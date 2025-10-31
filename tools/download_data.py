"""Script to download datasets."""

import argparse
import zipfile
import os
import shutil
import urllib.request


URLS = {
    'ecl_small': 'https://example.com/ecl_small.zip',  # replace with real mirrors
}


def download_dataset(dataset_name, subset='full', output_dir='data/raw'):
    """Download a dataset by name."""
    print(f"Downloading {dataset_name} (subset: {subset}) to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # TODO: Implement per-dataset fetchers here
    key = f"{dataset_name}_{subset}"
    if key in URLS:
        url = URLS[key]
        # Download logic here
        print(f"Would download from {url}")
    else:
        print(f"URL not configured for {key}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, 
                   choices=['ecl', 'electricity', 'etth', 'ettm', 'gefcom2014_solar', 'iso_pjm'])
    p.add_argument('--subset', default='full')
    p.add_argument('--output', default='data/raw')
    args = p.parse_args()
    
    download_dataset(args.dataset, args.subset, args.output)
