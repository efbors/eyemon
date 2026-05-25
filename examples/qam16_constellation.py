# examples/qam16_constellation.py
import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
import argparse
from pathlib import Path
import yaml
import numpy as np

from eyemon.eyemon import Eyemon


def main() -> None:
    """
    Main execution loop for the QAM16 constellation visualization example.
    """
    # Parse CLI Arguments
    parser = argparse.ArgumentParser(description="QAM16 constellation simulation")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to the experiment YAML config file")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()

    with open(config_path, 'r') as f:
        raw_yaml = f.read()
        expanded_yaml = os.path.expandvars(raw_yaml)  # Expand all env vars
        config = yaml.safe_load(expanded_yaml)

    print("Generating synthetic QAM16 data...")

    # Generate random symbols on the odd integer grid (-3, -1, 1, 3)
    grid_points = np.array([-3.0, -1.0, 1.0, 3.0])
    num_symbols = 300000

    i_symbols = np.random.choice(grid_points, size=num_symbols)
    q_symbols = np.random.choice(grid_points, size=num_symbols)

    # Combine into a complex signal array with additive white Gaussian noise
    rx_qam = i_symbols + 1j * q_symbols
    noise = np.random.normal(0, 0.08, num_symbols) + 1j * np.random.normal(0, 0.08, num_symbols)
    rx_qam += noise

    print("Launching Eyemon...")
    probe = Eyemon(config)

    loop = True  # True -> continuously loop through the experiment
    start_index = 0

    # Block matrix mapping dimensions (64 rows of 32 complex constellation points)
    nrow = 64
    seg_size = 32

    probe.animate_sweep(
        name='rx_qam',
        loop=loop,
        signal=rx_qam,
        seg_size=seg_size,
        nrow=nrow,
        fps=30,
        loop_duration=10,
        gain=100.0,
        start_index=start_index,
        ext_samples=0
    )


if __name__ == "__main__":
    main()