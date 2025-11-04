"""Script to download datasets."""

import argparse
import zipfile
import os
import urllib.request
import urllib.error


# Dataset download URLs
# ETT (Electricity Transformer Temperature) datasets only
URLS = {
    # ETTh1 and ETTh2 (hourly) - Using Hugging Face
    'etth_full': 'https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTh1.csv',
    'etth_small': 'https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTh1.csv',
    'etth1': 'https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTh1.csv',
    'etth2': 'https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTh2.csv',
    
    # ETTm1 and ETTm2 (15-minute) - Using Hugging Face
    'ettm_full': 'https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTm1.csv',
    'ettm_small': 'https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTm1.csv',
    'ettm1': 'https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTm1.csv',
    'ettm2': 'https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTm2.csv',
}


def download_file(url, output_path, show_progress=True):
    """Download a file from URL to output path."""
    try:
        if show_progress:
            def progress_hook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                if percent % 10 == 0:
                    print(f"\rDownloading... {percent}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, output_path, progress_hook)
            print()  # New line after progress
        else:
            urllib.request.urlretrieve(url, output_path)
        
        print(f"Successfully downloaded to {output_path}")
        return True
    except urllib.error.URLError as e:
        print(f"Error downloading from {url}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def extract_zip(zip_path, extract_dir):
    """Extract a zip file to a directory."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted {zip_path} to {extract_dir}")
        return True
    except zipfile.BadZipFile:
        print(f"Warning: {zip_path} is not a valid zip file, skipping extraction")
        return False
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def download_dataset(dataset_name, subset='full', output_dir='data/raw', force=False):
    """Download a dataset by name."""
    print(f"Downloading {dataset_name} (subset: {subset}) to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to find the URL
    key = f"{dataset_name}_{subset}"
    url = None
    
    if key in URLS:
        url = URLS[key]
    elif dataset_name in URLS:
        url = URLS[dataset_name]
    else:
        # Try alternative keys
        alt_keys = [
            f"{dataset_name}_full",
            f"{dataset_name}_small",
            dataset_name
        ]
        for alt_key in alt_keys:
            if alt_key in URLS:
                url = URLS[alt_key]
                break
    
    if not url:
        print(f"Error: URL not configured for dataset '{dataset_name}' with subset '{subset}'")
        print(f"Available keys: {list(URLS.keys())}")
        return False
    
    # Determine output filename
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Handle different URL types
    if 'kaggle.com' in url:
        print(f"\nNote: {dataset_name} is available on Kaggle.")
        print(f"Please download manually from: {url}")
        print("Or use Kaggle API: kaggle datasets download -d <dataset-id>")
        return False
    
    if 'pjm.com' in url:
        print(f"\nNote: {dataset_name} requires manual download from PJM website.")
        print(f"Visit: {url}")
        print("PJM data can be accessed via their Data Miner portal.")
        return False
    
    # Extract filename from URL
    filename = os.path.basename(url).split('?')[0]
    if not filename or filename == '':
        filename = f"{dataset_name}.csv"
    
    output_path = os.path.join(dataset_dir, filename)
    
    # Check if file already exists
    if os.path.exists(output_path) and not force:
        print(f"File already exists at {output_path}. Use --force to re-download.")
        return True
    
    print(f"Downloading from: {url}")
    success = download_file(url, output_path)
    
    
    if success and filename.endswith('.zip'):
        # Extract zip file
        extract_dir = dataset_dir
        extract_zip(output_path, extract_dir)
        # Optionally remove zip file after extraction
        # os.remove(output_path)
    
    return success


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Download ETT (Electricity Transformer Temperature) datasets')
    p.add_argument('--dataset', required=True, 
                   choices=['etth', 'ettm', 'etth1', 'etth2', 'ettm1', 'ettm2'],
                   help='Dataset name to download (etth or ettm)')
    p.add_argument('--subset', default='full', choices=['full', 'small'],
                   help='Dataset subset (full or small)')
    p.add_argument('--output', default='data/raw',
                   help='Output directory for downloaded data')
    p.add_argument('--force', action='store_true',
                   help='Force re-download even if file exists')
    args = p.parse_args()
    
    download_dataset(args.dataset, args.subset, args.output, args.force)
