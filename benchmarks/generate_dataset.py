import argparse
import csv
import random
import string
import time
from pathlib import Path
from typing import Any

# Constants for Magic Numbers
MILLION = 1_000_000
THOUSAND = 1_000

# Default Configuration
DEFAULT_ROWS = 10 * MILLION
DEFAULT_BATCH = 100 * THOUSAND


def format_row_count(rows: int) -> str:
    """
    Converts a number like 10,000,000 into '10m' or 500,000 into '500k'.
    """
    if rows >= MILLION:
        return f"{rows // MILLION}M"
    elif rows >= THOUSAND:
        return f"{rows // THOUSAND}K"
    return str(rows)


def generate_string(length: int = 10) -> str:
    """Generates a random string."""
    return "".join(random.choices(string.ascii_letters, k=length))


def generate_batch(batch_size: int) -> list[list[Any]]:
    """
    Generates a batch of rows with mixed types:
        - id: (Almost) Unique Integer
        - category: String (Low cardinality)
        - value: float
        - description: String (high cardinality)
        - status: String (with Nulls)
    """
    data = []
    for _ in range(batch_size):
        row = [
            random.randint(1, 1_000_000_000),
            random.choice(["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]),
            random.uniform(0.0, 1000.0),
            generate_string(15),
            random.choice(["Active", "Inactive", None]),
        ]
        data.append(row)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for Netra-Profiler benchmarks."
    )
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_ROWS, help="Number of rows to generate (default: 10M)"
    )
    parser.add_argument(
        "--batch", type=int, default=DEFAULT_BATCH, help="Batch size for writing (default: 100k)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Custom filename. If omitted, defaults to 'dataset_{N}m|k.csv'",
    )

    args = parser.parse_args()

    # Determine Filename
    if args.filename:
        filename = args.filename
        # Safety: We ensure user didn't try to traverse directories (e.g. ../../root.csv)
        if ".." in filename or "/" in filename or "\\" in filename:
            raise ValueError(
                "Filename must contain path separators. Files are saved to benchmarks/data/ only."
            )
    else:
        filename = f"dataset_{format_row_count(args.rows)}.csv"

    # We lock output to benchmarks/data
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    print(f"Generating {args.rows:,} ({format_row_count(args.rows)}) rows of synthetic data...")
    print(f"Target: {output_path}")

    start_time = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "value", "description", "status"])

        rows_written = 0
        while rows_written < args.rows:
            # Handle the final batch (Tail logic)
            current_batch_size = min(args.batch, args.rows - rows_written)

            batch = generate_batch(current_batch_size)
            writer.writerows(batch)
            rows_written += current_batch_size

            if rows_written % MILLION == 0:
                print(f"   Written {format_row_count(rows_written)} rows...")

    elapsed_time = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\nDone! Generated {format_row_count(args.rows)} rows in {elapsed_time:.2f} seconds")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Throughput: {rows_written / elapsed_time:.0f} rows/sec")


if __name__ == "__main__":
    main()
