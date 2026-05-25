# EyeMon: Hardware-Accelerated Signal Visualization

EyeMon is a high-performance, GPU-accelerated signal monitoring tool designed for visualizing complex multi-level waveforms and multi-level quadrature constellations. 

<table>
  <tr>
    <td width="48%" align="center" style="padding-right: 15px; border: none;">
      <h3>PAM4 Eye Diagram (Waveform)</h3>
      <video src="assets/pam4_fixed.mp4" autoplay loop muted playsinline width="100%"></video>
    </td>
    <td width="48%" align="center" style="padding-left: 15px; border: none;">
      <h3>16-QAM Constellation</h3>
      <video src="assets/qam16_fixed.mp4" autoplay loop muted playsinline width="100%"></video>
    </td>
  </tr>
</table>

---

## Design Philosophy

### The Tektronix P31 Phosphor Aesthetic
Modern digital plots often lack the intuitive depth of legacy analog 
lab equipment. EyeMon was deliberately designed to recreate the 
classic, highly responsive look and feel of the **P31 Phosphor** found 
in vintage Tektronix oscilloscopes. 

In analog scopes, the electron beam striking the phosphor screen 
created natural intensity grading—areas where the signal crossed 
frequently glowed brighter, while rare outliers faded into the 
dark background. EyeMon replicates this persistence and intensity 
grading mathematically. 
It allows the human eye to instantly distinguish between the 
deterministic core of a signal eye/constellation and the probabilistic 
noise floor or rare transient jitter.

### Multi-Platform GPU Acceleration via PyTorch
Generating high-resolution, real-time 2D histograms and applying 
phosphor decay persistence requires large matrix operations that 
may bottleneck a CPU. 

**Native support for NVIDIA, ROCm (AMD), or Metal (Apple Silicon)**.

EyeMon leverages **PyTorch (`torch`)** as its mathematical backend.
PyTorch provides a highly optimized, hardware-agnostic tensor ecosystem.
By utilizing PyTorch's native tensor math functions, 
EyeMon automatically detects and utilizes the system's available 
GPU compute.

---

## Installation

Requires Python 3.12+ installed. It is highly recommended to use a 
virtual environment.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/efbors/eyemon.git](https://github.com/efbors/eyemon.git)
   cd eyemon
   ```

2. **Install the dependencies:**
   EyeMon uses `pyproject.toml` for dependency management. It is 
   highly recommended to visit [pytorch.org](https://pytorch.org/) and install 
   your hardware-specific PyTorch version (CUDA/ROCm) *before* running the 
   general installation command below.
   ```bash
   pip install .
   ```

---

## Usage

EyeMon is designed to be highly flexible for both post-processing
and live hardware-in-the-loop development. It operates in two primary modes:

### Mode 1: Animation Mode (Post-Processing)
Use this mode when you have a pre-captured, long-duration
signal (e.g., loaded from a `.csv` or `.npy`) and you want 
to animate it to observe time-varying effects like thermal 
drift or slowly accumulating jitter.

```python
from eyemon import ScopeViewer

# Load your full signal array
signal_data = load_my_signal("trace_capture.npy")

# Initialize the viewer in animation mode
scope = ScopeViewer(mode="animate", phosphor="P31")

# Animate the entire trace with a specific window size
scope.animate_trace(signal_data, window_size=2048, step=512)
```

### Mode 2: Real-Time Mode (Tuning & Development)
This mode acts as a live software instrument. It is entirely 
frame-based, making it ideal for real-time DSP development, 
ML model training loops, or live hardware telemetry.

```python
from eyemon import ScopeViewer

# Initialize the viewer in real-time mode
scope = ScopeViewer(mode="realtime", phosphor="P31", decay_rate=0.95)

# Example of an active tuning or inference loop
while system_is_running:
    # Grab a live chunk of data from your SDR, ADC, or ML output
    live_chunk = get_hardware_buffer()
    
    # Push the frame to the scope for immediate rendering
    scope.push_frame(live_chunk)
```

### Credits:
scope inspired by 
https://github.com/RandomDude4/PhosPe