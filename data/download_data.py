"""
download_data.py — Download the Customer Support on Twitter dataset from Kaggle.

Usage:
    python data/download_data.py

Requires either:
  - Kaggle API credentials (~/.kaggle/kaggle.json)
  - OR manual download: place twcs.csv in data/raw/

Citation:
  Dataset: "Customer Support on Twitter" by Thought Vector
  Source: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
"""

import os
import sys
import subprocess
from pathlib import Path


def download_from_kaggle(output_dir: Path) -> bool:
    """Attempt to download using Kaggle API."""
    try:
        import kaggle  # noqa: F401
        print("[INFO] Downloading dataset via Kaggle API...")
        subprocess.run(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", "thoughtvector/customer-support-on-twitter",
                "-p", str(output_dir),
                "--unzip"
            ],
            check=True
        )
        print(f"[OK] Dataset downloaded to {output_dir}")
        return True
    except (ImportError, subprocess.CalledProcessError, Exception) as e:
        print(f"[WARN] Kaggle API download failed: {e}")
        return False


def verify_data(output_dir: Path) -> bool:
    """Check if twcs.csv exists in the output directory."""
    csv_path = output_dir / "twcs.csv"
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Found twcs.csv ({size_mb:.1f} MB)")
        return True
    return False


def main():
    # Project root is parent of data/
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    if verify_data(raw_dir):
        print("[INFO] Data already downloaded. Skipping.")
        return

    # Try Kaggle API
    if download_from_kaggle(raw_dir):
        if verify_data(raw_dir):
            return

    # Fallback: manual instructions
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("1. Go to: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
    print("2. Click 'Download' (you'll need a Kaggle account)")
    print("3. Unzip and place 'twcs.csv' in:")
    print(f"   {raw_dir}")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()
