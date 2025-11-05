"""Script to generate leaderboard of model results."""

import argparse
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import sys


def load_results_from_experiment(exp_dir: Path) -> Dict[str, Any]:
    """Load results from an experiment directory.
    
    Looks for:
    - results.json (from training)
    - results_{split}.json (from eval.py, e.g., results_test.json, results_val.json)
    
    Returns dict with model, dataset, split (if available), and metrics.
    """
    exp_dir = Path(exp_dir)
    
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory not found: {exp_dir}")
    
    # Try to find results.json or results_*.json files
    results_files = []
    
    # Check for results.json (from training)
    results_json = exp_dir / "results.json"
    if results_json.exists():
        results_files.append(results_json)
    
    # Check for results_{split}.json files (from eval.py)
    for results_file in exp_dir.glob("results_*.json"):
        results_files.append(results_file)
    
    if not results_files:
        raise ValueError(f"No results.json or results_*.json files found in {exp_dir}")
    
    # Load all results files and combine them
    all_results = []
    for results_file in results_files:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract split from filename if it's results_{split}.json
        split = None
        if results_file.stem.startswith('results_'):
            split = results_file.stem.replace('results_', '')
        
        result = {
            'model': data.get('model', 'unknown'),
            'dataset': data.get('dataset', 'unknown'),
            'split': split or data.get('split', 'val'),  # Default to 'val' if not specified
            'exp_dir': str(exp_dir),
            'exp_name': exp_dir.name,
        }
        
        # Add all metrics
        metrics = data.get('metrics', {})
        for metric_name, metric_value in metrics.items():
            result[metric_name] = metric_value
        
        all_results.append(result)
    
    # If multiple splits found, return the one with 'test' split, otherwise first one
    if len(all_results) > 1:
        test_result = next((r for r in all_results if r['split'] == 'test'), None)
        if test_result:
            return test_result
    
    return all_results[0]


def make_leaderboard(exp_dirs: List[str], output_path: str = None, sort_by: str = 'MAE') -> pd.DataFrame:
    """Create leaderboard from multiple experiment directories.
    
    Args:
        exp_dirs: List of experiment directory paths
        output_path: Optional path to save CSV file
        sort_by: Metric to sort by (default: 'MAE')
    
    Returns:
        DataFrame with leaderboard
    """
    results = []
    
    for exp_dir_str in exp_dirs:
        try:
            result = load_results_from_experiment(exp_dir_str)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load results from {exp_dir_str}: {e}", file=sys.stderr)
            continue
    
    if not results:
        raise ValueError("No valid results found in any experiment directory")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns: Model, Dataset, Split, then metrics, then other info
    metric_cols = [col for col in df.columns if col not in ['model', 'dataset', 'split', 'exp_dir', 'exp_name']]
    other_cols = ['exp_dir', 'exp_name'] if 'exp_dir' in df.columns else []
    
    # Build column order
    column_order = ['model', 'dataset', 'split'] + sorted(metric_cols) + other_cols
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    
    df = df[column_order]
    
    # Rename columns for better display
    df = df.rename(columns={
        'model': 'Model',
        'dataset': 'Dataset',
        'split': 'Split',
        'exp_dir': 'Experiment Dir',
        'exp_name': 'Experiment Name'
    })
    
    # Sort by specified metric (lower is better for most metrics)
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=True)
    elif 'MAE' in df.columns:
        df = df.sort_values(by='MAE', ascending=True)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Add rank column
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    # Display leaderboard
    print("\n" + "="*80)
    print("LEADERBOARD")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save to CSV if output path specified
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nLeaderboard saved to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate leaderboard from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare multiple experiments
  python tools/make_leaderboard.py runs/exp1 runs/exp2 runs/exp3
  
  # Compare all experiments in runs directory
  python tools/make_leaderboard.py runs/*
  
  # Save to CSV
  python tools/make_leaderboard.py runs/exp1 runs/exp2 --output leaderboard.csv
  
  # Sort by different metric
  python tools/make_leaderboard.py runs/exp1 runs/exp2 --sort-by RMSE
        """
    )
    parser.add_argument("exp_dirs", nargs="+", type=str,
                       help="One or more experiment directory paths")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output CSV file path (default: don't save)")
    parser.add_argument("--sort-by", type=str, default="MAE",
                       help="Metric to sort by (default: MAE)")
    
    args = parser.parse_args()
    
    try:
        make_leaderboard(args.exp_dirs, args.output, args.sort_by)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
