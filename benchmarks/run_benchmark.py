import argparse
import time
from pathlib import Path

import polars as pl

from netra_profiler import Profiler, __version__


def main() -> None:
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(
        description="Run Netra-Profiler benchmark on a specific dataset."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="dataset_10M.csv",
        help="Filename in benchmarks/data/ to profile (supports .csv and .parquet)",
    )
    args = parser.parse_args()

    # 2. Locate File
    data_dir = Path(__file__).parent / "data"
    data_file = data_dir / args.filename

    if not data_file.exists():
        print(f"Error: File not found at {data_file}")
        print("   Available files:")
        for f in data_dir.glob("*"):
            print(f"   - {f.name}")
        return

    # Calculate size in bytes first for precision
    file_size_bytes = data_file.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_bytes / (1024 * 1024 * 1024)

    print(f"Starting Benchmark on {data_file.name} ({file_size_mb:.2f} MB)")
    print(f"   Netra-Profiler Version: v{__version__}")

    # 3. Smart Load (CSV vs Parquet)
    # We explicitly choose the right scanner based on extension
    if data_file.suffix.lower() == ".parquet":
        lf = pl.scan_parquet(data_file)
    else:
        # Default to CSV
        lf = pl.scan_csv(data_file)

    # 4. Execution
    start_time = time.time()

    profiler = Profiler(lf)
    profile = profiler.profile()

    elapsed = time.time() - start_time

    # 5. Results
    rows_per_sec = profile["table_row_count"] / elapsed if elapsed > 0 else 0
    gb_per_sec = file_size_gb / elapsed if elapsed > 0 else 0

    print("\nProfiling Complete!")
    print(f"   Time Taken:   {elapsed:.2f} seconds")
    print(f"   Throughput:   {rows_per_sec:,.0f} rows/sec")
    print(f"   Bandwidth:    {gb_per_sec:.2f} GB/s")
    print("-" * 40)
    print("Key Stats Verified:")
    print(f"   • Row Count:   {profile['table_row_count']:,}")
    if "value_mean" in profile:
        print(f"   • Value Mean:  {profile['value_mean']:.2f}")
    if "category_max" in profile:
        print(f"   • Category Max: {profile['category_max']}")
    if "description_len_max" in profile:
        print(f"   • Desc Max Length: {profile['description_len_max']}")


if __name__ == "__main__":
    main()
