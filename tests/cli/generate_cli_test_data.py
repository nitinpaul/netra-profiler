from pathlib import Path

import numpy as np
import polars as pl

NULL_RATIO = 0.99


def generate_cli_test_data() -> None:
    print("Synthesizing Netra CLI Test Data...")

    # Resolve paths so it always writes to tests/data/ relative to this script
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "cli_testbench.csv"

    dataset_size = 100_000
    np.random.seed(42)  # For reproducibility

    # 1. Distributions (To test our inline sparklines)
    normal_distribution = np.random.normal(loc=50, scale=10, size=dataset_size)
    uniform_distribution = np.random.uniform(low=0, high=100, size=dataset_size)
    skewed_distribution = np.random.exponential(scale=2.0, size=dataset_size)

    # Bimodal (combining two normal distributions)
    bimodal_1 = np.random.normal(loc=25, scale=5, size=dataset_size // 2)
    bimodal_2 = np.random.normal(loc=75, scale=5, size=dataset_size // 2)
    bimodal_distribution = np.concatenate([bimodal_1, bimodal_2])
    np.random.shuffle(bimodal_distribution)

    # 2. Correlations (To test our Multicollinearity Card)
    correlation_positive_target = normal_distribution * 3.5 + np.random.normal(0, 2, dataset_size)
    correlation_negative_target = -normal_distribution * 2.0 + np.random.normal(0, 2, dataset_size)

    # 3. Health Hazards (To test the Triage Card)
    high_nulls_warning = np.random.normal(0, 1, dataset_size)
    mask = np.random.rand(dataset_size) < NULL_RATIO
    high_nulls_warning[mask] = np.nan

    constant_warning = np.full(dataset_size, 42.0)

    # 4. String / Categorical Columns (To test Top-K Polymorphic column)
    country_categories = np.random.choice(
        ["NL", "DE", "BE", "FR", "UK"], size=dataset_size, p=[0.55, 0.20, 0.15, 0.08, 0.02]
    )

    messy_categories = np.random.choice(
        ["Valid_String", "", "   ", "N/A"], size=dataset_size, p=[0.85, 0.05, 0.05, 0.05]
    )

    unique_ids = [f"USR-{i:06d}" for i in range(dataset_size)]

    # Compile the Dataframe
    df = pl.DataFrame(
        {
            "normal_distribution": normal_distribution,
            "uniform_distribution": uniform_distribution,
            "skewed_distribution": skewed_distribution,
            "bimodal_distribution": bimodal_distribution,
            "correlation_positive_target": correlation_positive_target,
            "correlation_negative_target": correlation_negative_target,
            "high_nulls_warning": high_nulls_warning,
            "constant_warning": constant_warning,
            "country_categories": country_categories,
            "messy_categories": messy_categories,
            "unique_ids": unique_ids,
        }
    )

    df.write_csv(output_path)
    print(f"✔ Successfully generated {output_path.name} ({dataset_size} rows) in {data_dir.name}/")


if __name__ == "__main__":
    generate_cli_test_data()
