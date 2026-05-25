# examples/pam4_eye.py
import numpy as np
from scipy.ndimage import gaussian_filter1d
import argparse
import os
from pathlib import Path
import yaml

from eyemon.eyemon import Eyemon  # Assuming your main class is here


def main():
    # Parse CLI Arguments
    parser = argparse.ArgumentParser(description="200G DSP simulation")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to the experiment YAML config file")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()

    with open(config_path, 'r') as f:
        raw_yaml = f.read()
        expanded_yaml = os.path.expandvars(raw_yaml)  # Expand all env vars
        config = yaml.safe_load(expanded_yaml)

    print("Generating synthetic PAM4 data...")

    # Generate random PAM4 symbols (-3, -1, 1, 3)
    symbols = np.random.choice([-3.0, -1.0, 1.0, 3.0], size=5000)

    # Use Zero-Order Hold (ZOH) to create rectangular pulses
    os_factor = 32
    tx_analog = np.repeat(symbols, os_factor)

    # Simulate the channel (Gaussian acts as dispersion/ISI)
    rx_analog = gaussian_filter1d(tx_analog, sigma=3.0)
    rx_analog += np.random.normal(0, 0.05, len(rx_analog))  # Add AWGN

    print("Launching Eyemon...")
    probe = Eyemon(config)

    loop = True # True-> continuously loop thru the experiment
    start_index = 1000  # start plotting from sample index 1000
    # Run the hardware-accelerated sweep

    probe.animate_sweep(
        name='rx_pam4',
        loop=loop,
        signal=rx_analog,
        seg_size=os_factor,
        nrow=64,
        fps=30,
        loop_duration=10,
        gain=100.0,
        start_index=start_index,
        ext_samples=int(1.0 * os_factor)
    )


if __name__ == "__main__":
    main()
