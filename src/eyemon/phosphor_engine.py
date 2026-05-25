import torch
import torch.nn.functional as F


class PhosphorEngine:
    def __init__(self, config):
        self.config = config
        self.tools_cfg = config.get('tools', {})

        # Automatically route to the best available hardware
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.win_resolutin_x = self.config['tools']['win_resolutin_x']
        self.plot_resolution_y = self.config['tools']['plot_resolution_y']

        # =======================================================================
        # Kernel Initialization (Directly on GPU)
        # =======================================================================
        self.enable_phosphor_bloom = self.tools_cfg.get('enable_phosphor_bloom', True)
        self.p31_grid_size_pix = self.tools_cfg.get('p31_grid_size_pix', 31)
        self.core_gaussian_sd = self.tools_cfg.get('core_gaussian_sd', 1.0)
        self.halo_gaussian_sd = self.tools_cfg.get('halo_gaussian_sd', 4.0)

        center = self.p31_grid_size_pix // 2
        grid_1d = torch.arange(-center, center + 1, dtype=torch.float32, device=self.device)
        x, y = torch.meshgrid(grid_1d, grid_1d, indexing='xy')
        r = torch.sqrt(x ** 2 + y ** 2)

        E = torch.exp(-(r ** 2) / (2.0 * self.core_gaussian_sd ** 2))
        if E.sum() > 0: E /= E.sum()

        # Note: Renamed 'F' to 'F_kernel' to avoid conflict with torch.nn.functional
        F_kernel = torch.exp(-r / (self.halo_gaussian_sd * 0.2))
        if F_kernel.sum() > 0: F_kernel /= F_kernel.sum()

        G = torch.exp(-r / (self.p31_grid_size_pix / 16.0))
        if G.sum() > 0: G /= G.sum()

        dot_intens = [1.0, 0.4, 1.0]
        self.phosphor_kernel = (E * dot_intens[0]) + (F_kernel * dot_intens[1]) + (G * dot_intens[2])
        self.phosphor_kernel /= self.phosphor_kernel.sum()

        # PyTorch conv2d expects shape (out_channels, in_channels, H, W)
        self.phosphor_kernel = self.phosphor_kernel.unsqueeze(0).unsqueeze(0)

        # =======================================================================
        # LUT Initialization (Directly on GPU)
        # =======================================================================
        rgb_base = torch.tensor([0.2, 1.0, 0.76], dtype=torch.float32, device=self.device)
        map_index = torch.linspace(0, 1, 1024, device=self.device).unsqueeze(1)
        multfactor = 10.0

        cmap_colors = torch.tanh(map_index * rgb_base * multfactor)
        cmap_colors = torch.clamp(cmap_colors, 0.0, 1.0)

        alpha_channel = torch.ones((1024, 1), dtype=torch.float32, device=self.device)
        self.cmap_lut = torch.hstack((cmap_colors, alpha_channel))

    def process_and_render(self, frame_2d_np, plot_type, window_gain):
        """
            Executes the entire optical pipeline on the GPU.
            Returns a 1D flat NumPy array ready for DearPyGui.
        """
        # Pull data into VRAM without forcing float32 immediately to preserve complex types
        frame_2d = torch.from_numpy(frame_2d_np).to(self.device)
        x_res = self.win_resolutin_x
        y_res = self.plot_resolution_y

        if plot_type == 'constellation':
            # Extract spatial IQ coordinates from the complex numbers
            x_coords = torch.real(frame_2d)
            y_coords = torch.imag(frame_2d)

            # Scale coordinates to fit the grid resolution bounds
            scaled_x = x_coords * window_gain
            scaled_y = y_coords * window_gain

            # Map directly to discrete screen pixel coordinates centered on the viewport
            x_indices = torch.clamp(torch.round(scaled_x + x_res / 2.0), 0, x_res - 1).to(torch.int32)
            y_indices = torch.clamp(torch.round(scaled_y + y_res / 2.0), 0, y_res - 1).to(torch.int32)

            # Vectorized Scatter for independent discrete impacts
            flat_indices = y_indices.flatten() * x_res + x_indices.flatten()
            flat_canvas = torch.bincount(flat_indices.to(torch.int64), minlength=y_res * x_res)
            canvas = flat_canvas.reshape((y_res, x_res)).to(torch.float32)
        else:
            # Enforce float32 representation for chronological waveform tracing
            frame_2d = frame_2d.to(torch.float32)
            windows_per_frame, original_length = frame_2d.shape

            # Sinc Interpolation (GPU FFT)
            if original_length != x_res:
                fft_data = torch.fft.fft(frame_2d, dim=1)
                fft_padded = torch.zeros((windows_per_frame, x_res), dtype=torch.complex128, device=self.device)
                half_len = original_length // 2

                fft_padded[:, :half_len] = fft_data[:, :half_len]

                if original_length % 2 == 0:
                    fft_padded[:, half_len] = fft_data[:, half_len] / 2.0
                    fft_padded[:, x_res - half_len] = fft_data[:, half_len] / 2.0
                    fft_padded[:, x_res - half_len + 1:] = fft_data[:, half_len + 1:]
                else:
                    fft_padded[:, x_res - half_len:] = fft_data[:, half_len + 1:]

                upsampled_y = torch.real(torch.fft.ifft(fft_padded, dim=1)) * (x_res / original_length)
            else:
                upsampled_y = frame_2d

            # Y-Axis Scaling & Distance Intensity
            scaled_y = upsampled_y * window_gain
            y_max = y_res / 2.0
            frame_pix = torch.clamp(scaled_y, -y_max, y_max).to(torch.float16)
            dy = torch.diff(frame_pix, dim=1)
            # A 2-element tuple tells PyTorch to only pad the last dimension (left=0, right=1)
            dy = F.pad(dy, (0, 1), mode='replicate')
            distance = torch.sqrt(1.0 + dy ** 2)
            frame_intensity = (1.0 / distance).to(torch.float16)

            # Vectorized Scatter (torch.bincount)
            y_indices = torch.clamp(torch.round(frame_pix + y_res / 2.0), 0, y_res - 1).to(torch.int32)
            x_indices = torch.tile(torch.arange(x_res, device=self.device), (windows_per_frame, 1)).to(torch.int32)

            flat_indices = y_indices.flatten() * x_res + x_indices.flatten()
            # PyTorch bincount strictly requires int64 for indices and float32/float64 for weights
            flat_canvas = torch.bincount(flat_indices.to(torch.int64),
                                         weights=frame_intensity.flatten().to(torch.float32),
                                         minlength=y_res * x_res)
            canvas = flat_canvas.reshape((y_res, x_res))

        # Phosphor Bloom (GPU Spatial Convolution)
        if self.enable_phosphor_bloom:
            # Reshape (H, W) to (Batch, Channels, H, W) for F.conv2d
            canvas = canvas.unsqueeze(0).unsqueeze(0)
            canvas = F.conv2d(canvas, self.phosphor_kernel, padding='same')
            canvas = canvas.squeeze(0).squeeze(0)

        # Gamma Correction & Colormap LUT Mapping
        vmax = torch.max(canvas) * 0.4
        if vmax > 0:
            norm_canvas = torch.clamp(canvas / vmax, 0.0, 1.0)
        else:
            norm_canvas = canvas

        norm_canvas = norm_canvas ** 0.8
        # .to() ensures the type conversion happens directly on the compute cores
        lut_indices = torch.clamp((norm_canvas * 1023).to(torch.int32), 0, 1023)

        # PyTorch array indexing requires int64 (long)
        rgba_image = self.cmap_lut[lut_indices.to(torch.int64)]

        # Push flattened texture back to System RAM for DPG
        return rgba_image.flatten().cpu().numpy()
