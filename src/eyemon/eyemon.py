"""
eyemon.py

Core hardware-accelerated monitoring probe for real-time DSP simulations.
Handles data staging, multi-process queue dispatching, and framing logic.
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided
import multiprocessing
from typing import Dict, Any, Optional
import time


class Eyemon:
    """
        Main interface for staging and dispatching DSP arrays to the rendering backend.
        """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        self.tools_cfg = config.get('tools', {})
        self.enable_gui = self.tools_cfg.get('enable_gui', False)

        # Internal staging area for accumulating data before a flush
        self._staging_buffer = {}
        self._last_flush_time = 0.0

        if self.enable_gui:
            queue_size = self.tools_cfg.get('queue_max_size', 5)
            self.queue = multiprocessing.Queue(maxsize=queue_size)

            # Import GUI here to avoid circular imports if they live in different files
            from eyemon.dsp_gui import DspGui

            self.gui_process = multiprocessing.Process(
                target=DspGui.launch,
                args=(self.config, self.queue)
            )
            self.gui_process.start()
        else:
            self.queue = None
            self.gui_process = None
            # TODO: Initialize your HDF5 / Cloud / Datalake logger here
            print("DspProbe: GUI disabled. Running in headless data-logging mode.")

    def clear(self, max_frames: int, fps: int, loop: bool = False) -> None:
        """
        Starts a new probing session and configures the viewer timeline.
        """

        self.max_frames = max_frames
        self.fps = fps
        self.loop = loop

        # Reset counters and internal state
        self.frame_index = 0
        self._staging_buffer.clear()

        # Reset the pacing clock
        self._last_flush_time = time.time()

    def plot(self, name: str, frame_index: int, input_signal: np.ndarray,
             gain: float,
             start_index: int, nrow: int, seg_size: int, ext_samples: int) -> None:
        """
        Adds a frame to the plot window, automatically branching processing
        based on real or complex signal type.

        :param name: Identifier for the signal.
        :param frame_index: Current frame number in the sequence.
        :param input_signal: Raw 1D array of the signal.
        :param gain: Vertical scaling multiplier.
        :param start_index: Array offset for the beginning of this frame.
        :param nrow: Number of vertical traces to stack.
        :param seg_size: Number of samples per symbol/trace.
        :param ext_samples: Boundary extension for edge overlaps.
        """

        if 'plot' not in self._staging_buffer:
            self._staging_buffer['plot'] = {}

        if np.iscomplexobj(input_signal):
            # Extract the raw block data for constellation coordinate mapping
            total_samples = nrow * seg_size
            slice_data = input_signal[start_index:start_index + total_samples]

            # Safeguard boundary constraints
            if len(slice_data) < total_samples:
                slice_data = np.pad(slice_data, (0, total_samples - len(slice_data)), mode='constant')

            frame_2d = slice_data.reshape(nrow, seg_size)

            # Stage the data and its metadata using the specific signal name
            self._staging_buffer['plot'][name] = {
                'frame_2d': frame_2d,
                'frame_index': frame_index,
                'plot_type': 'constellation',
                'gain': gain
            }
        else:
            # Extract overlapping time windows using stride geometry
            frame_2d = self._signal_frame(input_signal, start_index, nrow,
                                          seg_size, ext_samples)

            # Stage the data and its metadata using the specific signal name
            self._staging_buffer['plot'][name] = {
                'frame_2d': frame_2d,
                'frame_index': frame_index,
                'plot_type': 'waveform',
                'gain': gain
            }


    def trace(self, vals: np.ndarray, ymin: float, ymax: float) -> None:
        pass

    def spec(self) -> None:
        pass

    def scalar(self, index: int, name: str, value: float) -> None:
        pass

    def flush(self, frame_index: int) -> None:
        """
        Packages the staged data, enforces the simulation frame rate,
        dispatches the payload, and resets for the next frame.
        """
        # Dynamic Hardware Pacing
        current_time = time.time()
        elapsed = current_time - self._last_flush_time
        target_elapsed = 1.0 / self.fps

        # Only sleep if the math took LESS time than the frame budget
        sleep_time = target_elapsed - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Record the time AFTER the sleep as the new baseline for the next frame
        self._last_flush_time = time.time()

        # Package the Payload
        # Pack the frame index and the structured dictionary we built via plot(), trace(), etc.
        payload = {
            'frame_index': frame_index,
            'data': self._staging_buffer
        }

        # Dispatch the Payload
        if self.enable_gui and self.queue is not None:
            try:
                # block=True (default) creates natural backpressure.
                # If the GUI is slow, the simulation will pause here and wait.
                self.queue.put(payload)
            except Exception as e:
                print(f"DspProbe: Failed to push to queue: {e}")
        else:
            # TODO: Headless mode: Insert datalake / file saving logic here
            pass

        # Reset the Staging Area
        # Assign a new dictionary rather than calling .clear()
        self._staging_buffer = {}
        self.frame_index += 1

    def _signal_frame(self, signal_in: np.ndarray, start_index: int, windows_per_frame: int,
                      window_size: int, ext_samples: int) -> np.ndarray:
        """
        Extracts a 2D frame of overlapping windows using NumPy stride tricks.

        :param signal_in: 1D NumPy array representing the input signal.
        :param start_index: Starting index for the core frame.
        :param windows_per_frame: Number of rows in the resulting 2D frame.
        :param window_size: Core size of each window before extending.
        :param ext_samples: Number of overlapping samples added to left and right edges.
        :return: 2D NumPy array of shape (windows_per_frame, window_size + 2 * extension)
        """

        # Calculate the absolute start and end indices needed from the input array
        # The first row starts at: start_index - ext_samples
        # The last row ends at: start_index + (windows_per_frame * window_size) + ext_samples
        required_start = start_index - ext_samples
        required_end = start_index + (windows_per_frame * window_size) + ext_samples

        # Calculate necessary zero-padding if the required bounds fall outside signal_in
        pad_left = max(0, -required_start)
        pad_right = max(0, required_end - len(signal_in))

        # Safe bounds for slicing the valid part of the input signal
        slice_start = max(0, required_start)
        slice_end = min(len(signal_in), required_end)

        valid_data = signal_in[slice_start:slice_end]

        # Apply zero padding if beyond the boundaries (start or end of the signal)
        if pad_left > 0 or pad_right > 0:
            padded_data = np.pad(valid_data, (pad_left, pad_right), mode='constant', constant_values=0)
        else:
            padded_data = valid_data

        # Use NumPy stride tricks to construct the 2D array
        row_length = window_size + 2 * ext_samples
        shape = (windows_per_frame, row_length)

        # Each row advances by 'window_size' samples relative to the previous row
        strides = (window_size * padded_data.itemsize, padded_data.itemsize)

        # Create a copy of the strided view
        frame_2d = np.copy(as_strided(padded_data, shape=shape, strides=strides, writeable=False))

        # Edge Tapering (Custom Tukey Window)
        taper_len = int(.65 * window_size)

        # Generate the rising half of a Hann window: 0.5 * (1 - cos(pi * n / L))
        taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper_len) / taper_len))

        # Build the 1D flat window: [Rising Taper] + [Flat 1.0s] + [Falling Taper]
        custom_window = np.ones(row_length, dtype=np.float32)
        custom_window[:taper_len] = taper
        custom_window[-taper_len:] = taper[::-1]

        # Apply the window to all rows simultaneously via NumPy broadcasting
        frame_2d *= custom_window

        return frame_2d

    def animate_sweep(self, name: str, loop: bool, signal: np.ndarray, seg_size: int, nrow: int,
                      fps: int, loop_duration: float, gain: float, start_index: int, ext_samples: int) -> None:
        """
        Takes a fully computed 1D array and animates a sweep through it over time
        in the GUI, pausing the main simulation thread until the animation completes.
        """
        # If the GUI is disabled, just skip the animation entirely
        if not self.enable_gui:
            return

        ext_samples = int(np.round(ext_samples))
        nframes_to_show = int(loop_duration * fps)

        # Calculate the absolute maximum starting index
        max_safe_start = len(signal) - (nrow * seg_size) - ext_samples
        total_distance = max_safe_start - start_index

        if total_distance <= 0:
            print(f"Warning: Signal is too short to sweep. Displaying static frame.")
        else:

            chunk_size = nrow * seg_size
            total_chunks = total_distance // chunk_size

            frames_stride = total_chunks // nframes_to_show

            # Move by at least one frame
            if frames_stride == 0:
                frames_stride = 1

            advance_per_frame = frames_stride * chunk_size

            # Initialize the dashboard for this sweep (pass your new loop flag down!)
            self.clear(max_frames=nframes_to_show, fps=fps, loop=loop)

            current_index = start_index
            frame_index = 0
            try:
                while True:
                    self.plot(
                        name=name,
                        frame_index=frame_index,
                        input_signal=signal,
                        gain=gain,
                        start_index=current_index,
                        nrow=nrow,
                        seg_size=seg_size,
                        ext_samples=ext_samples
                    )

                    # Flush triggers the dynamic hardware pacing
                    self.flush(frame_index)

                    print(f"efb: frame_index={frame_index}")

                    current_index += advance_per_frame
                    frame_index += 1

                    # If NOT looping: break if hit the frame limit OR run out of data
                    if not loop and (frame_index >= nframes_to_show or current_index > max_safe_start):
                        break

                    #  If looping: wrap the array pointer and reset the P31 frame counter
                    if loop and current_index > max_safe_start:
                        current_index = start_index
                        frame_index = 0

            except (KeyboardInterrupt, BrokenPipeError, EOFError):
                # If pressing Escape kills the DearPyGui process, the queue will close.
                # This catches that broken pipe and exits cleanly without throwing a massive traceback.
                print("\n-- Sweep terminated by user.")
                return
