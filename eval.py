"""Evaluation script for time series forecasting models."""

import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description="Evaluate time series forecasting model")
    parser.add_argument("--config", type=str, default="configs/defaults.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Evaluation logic here
    print("Evaluation not implemented yet")


if __name__ == "__main__":
    main()

