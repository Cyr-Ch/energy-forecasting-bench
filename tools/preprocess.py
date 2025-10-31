"""Script to preprocess datasets."""

import argparse
import yaml


def preprocess_dataset(dataset_name, config_path):
    """Preprocess a dataset."""
    print(f"Preprocessing {dataset_name}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Preprocessing logic here
    print("Preprocessing not implemented yet")


def main():
    parser = argparse.ArgumentParser(description="Preprocess time series datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name to preprocess")
    parser.add_argument("--config", type=str,
                        help="Path to dataset config file")
    args = parser.parse_args()
    
    if args.config is None:
        args.config = f"configs/datasets/{args.dataset}.yaml"
    
    preprocess_dataset(args.dataset, args.config)


if __name__ == "__main__":
    main()

