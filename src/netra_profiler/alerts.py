from dataclasses import dataclass
from enum import Enum
from typing import Any


class AlertLevel(str, Enum):
    CRITICAL = "CRITICAL"  # Data is broken/unusable
    WARNING = "WARNING"  # Data is suspicious/requires attention
    INFO = "INFO"  # Optimization tip


@dataclass
class DiagnosticConfig:
    """Centralized configuration for diagnostic thresholds."""

    NULL_CRITICAL_THRESHOLD: float = 0.95
    NULL_WARNING_THRESHOLD: float = 0.50
    SKEW_THRESHOLD: float = 2.0
    ZERO_INFLATED_THRESHOLD: float = 0.10
    HIGH_CARDINALITY_THRESHOLD: int = 10_000
    HIGH_CORRELATION_THRESHOLD: float = 0.99
    ID_UNIQUENESS_THRESHOLD: float = 0.99
    MIN_ROWS_FOR_ID_CHECK: int = 100
    POSSIBLE_NUMERIC_CHECK_COUNT: int = 5


# Initialize default config
config = DiagnosticConfig()


@dataclass
class Alert:
    column: str
    alert_type: str  # e.g. "HIGH_NULLS"
    level: AlertLevel
    message: str
    value: float | None = None


class DiagnosticEngine:
    """
    Analyzes the profile dictionary and generates a list of Alerts.
    This is a pure logic layer: it does not query the data.
    """

    def __init__(self, profile: dict[str, Any]):
        self.profile = profile
        self.row_count = profile.get("table_row_count", 0)
        self.alerts: list[Alert] = []

    def run(self) -> list[Alert]:
        """Runs all diagnostic checks and returns the alerts."""
        if self.row_count == 0:
            return []

        # 1. Scalar Checks (Column by Column)

        # Extract unique column names from the keys
        # We iterate through keys to find column names based on patterns like "_null_count"
        # This is robust because our profile keys are structured: "{column_name}_{metric}"
        columns = set()
        for key in self.profile:
            if key.endswith("_null_count"):
                columns.add(key.replace("_null_count", ""))

        for col in columns:
            self._check_nulls(col)
            self._check_constant(col)
            self._check_cardinality(col)
            self._check_skew(col)
            self._check_zeros(col)
            self._check_possible_numeric(col)

        # 2. Global/Relationship Checks
        self._check_correlations()

        return self.alerts

    def _check_nulls(self, column_name: str) -> None:
        """
        Analyzes missing data.

        Alerts:
            - EMPTY_COLUMN (CRITICAL): Nulls > 95%. Likely useless.
            - HIGH_NULLS (CRITICAL): Nulls > 50%. Hard to impute.
        """
        null_count = self.profile.get(f"{column_name}_null_count", 0)
        null_percentage = null_count / self.row_count

        if null_percentage > config.NULL_CRITICAL_THRESHOLD:
            self.alerts.append(
                Alert(
                    column=column_name,
                    alert_type="EMPTY_COLUMN",
                    level=AlertLevel.CRITICAL,
                    message=(
                        f"Column is {null_percentage:.1%} empty. "
                        "It likely contains no useful information."
                    ),
                    value=null_percentage,
                )
            )
        elif null_percentage > config.NULL_WARNING_THRESHOLD:
            self.alerts.append(
                Alert(
                    column=column_name,
                    alert_type="HIGH_NULLS",
                    level=AlertLevel.CRITICAL,
                    message=f"Column is {null_percentage:.1%} empty. Imputation may be difficult.",
                    value=null_percentage,
                )
            )

    def _check_constant(self, column_name: str) -> None:
        """
        Analyzes value variance.

        Alerts:
            - CONSTANT (CRITICAL): Only 1 unique value. Adds no information.
            - ALL_DISTINCT (CRITICAL): Unique count ~= Row count. Likely an ID/PII.
        """
        n_unique = self.profile.get(f"{column_name}_n_unique")

        if n_unique == 1:
            self.alerts.append(
                Alert(
                    column=column_name,
                    alert_type="CONSTANT",
                    level=AlertLevel.CRITICAL,
                    message="Column has only 1 unique value. It adds no variance to the dataset.",
                    value=1.0,
                )
            )

        # Check for ALL_DISTINCT (ID columns)
        # We flag if the dataset is reasonably sized (> 100 rows)
        # and unique count is very close to row count (e.g. > 99%).
        if (
            n_unique
            and self.row_count > config.MIN_ROWS_FOR_ID_CHECK
            and n_unique > (self.row_count * config.ID_UNIQUENESS_THRESHOLD)
        ):
            # HLL can overshoot slightly, leading to >100% distinctness which looks buggy.
            # So we clamp the ratio for display to 100% max
            display_ratio = min(n_unique / self.row_count, 1.0)

            self.alerts.append(
                Alert(
                    column=column_name,
                    alert_type="ALL_DISTINCT",
                    level=AlertLevel.INFO,
                    message=(
                        f"Column is {display_ratio:.1%} distinct. Likely a Primary Key or ID."
                    ),
                    value=n_unique,
                )
            )

    def _check_cardinality(self, column_name: str) -> None:
        """
        Analyzes cardinality of string columns.

        Alerts:
            - HIGH_CARDINALITY (WARNING): > 10k unique values. Expensive for ML.
        """
        n_unique = self.profile.get(f"{column_name}_n_unique")
        dtype_hint = "String" if f"{column_name}_mean" not in self.profile else "Numeric"

        # Only flag strings (Numeric high cardinality is normal)
        if (
            dtype_hint == "String"
            and n_unique
            and n_unique > config.HIGH_CARDINALITY_THRESHOLD
            and n_unique < self.row_count
        ):
            self.alerts.append(
                Alert(
                    column=column_name,
                    alert_type="HIGH_CARDINALITY",
                    level=AlertLevel.WARNING,
                    message=f"High cardinality ({n_unique} unique values). Avoid One-Hot Encoding.",
                    value=n_unique,
                )
            )

    def _check_skew(self, column_name: str) -> None:
        """
        Analyzes distribution shape.

        Alerts:
            - SKEWED (WARNING): Skew > 2.0. Requires log-transformation.
        """
        skew = self.profile.get(f"{column_name}_skew")
        if skew is not None and abs(skew) > config.SKEW_THRESHOLD:
            self.alerts.append(
                Alert(
                    column=column_name,
                    alert_type="SKEWED",
                    level=AlertLevel.WARNING,
                    message=(
                        f"Distribution is highly skewed ({skew:.2f}). "
                        "Linear models may require transformation."
                    ),
                    value=skew,
                )
            )

    def _check_zeros(self, column_name: str) -> None:
        """
        Analyzes zero-inflation.

        Alerts:
            - ZERO_INFLATED (WARNING): > 10% zeros. Potential missing data proxy.
        """
        # We use Top-K to detect this cheaply.
        top_k = self.profile.get(f"{column_name}_top_k")
        if top_k and isinstance(top_k, list):
            for item in top_k:
                # Check if value is 0 (or 0.0)
                if item["value"] == 0 or item["value"] == 0.0:
                    zero_count = item["count"]
                    zero_percentage = zero_count / self.row_count

                    if zero_percentage > config.ZERO_INFLATED_THRESHOLD:
                        self.alerts.append(
                            Alert(
                                column=column_name,
                                alert_type="ZERO_INFLATED",
                                level=AlertLevel.WARNING,
                                message=(
                                    f"Column is {zero_percentage:.1%} zeros. "
                                    "Check if '0' represents missing data."
                                ),
                                value=zero_percentage,
                            )
                        )
                    break

    def _check_possible_numeric(self, column_name: str) -> None:
        """
        Heuristic check for string columns that contain numbers.

        Alerts:
            - POSSIBLE_NUMERIC (INFO): Recommendation to cast type.
        """
        # 1. Skip if already numeric
        if f"{column_name}_mean" in self.profile:
            return

        # We check the Top-K most frequent values. If the top 5 values
        # can all be cast to float, it is highly likely the column is numeric.
        # We use Top-K because checking every row is expensive (O(N)),
        # while Top-K is O(1) here.
        top_k = self.profile.get(f"{column_name}_top_k")

        # 2. Validate input structure
        if not top_k or not isinstance(top_k, list) or len(top_k) == 0:
            return

        # 3. Check the sample (Top 5)
        # We only look at values that are NOT None
        sample_values = [
            item["value"]
            for item in top_k[: config.POSSIBLE_NUMERIC_CHECK_COUNT]
            if item["value"] is not None
        ]

        if not sample_values:
            return

        # 4. Try to convert all sampled values
        try:
            # If ANY value fails conversion, the whole column is treated as String
            # This is strict but safe.
            for value in sample_values:
                float(value)

            # If we survived the loop, they are all numbers
            self.alerts.append(
                Alert(
                    column=column_name,
                    alert_type="POSSIBLE_NUMERIC",
                    level=AlertLevel.INFO,
                    message=("Top values look like numbers. Consider casting to Integer/Float."),
                    value=None,
                )
            )
        except ValueError:
            # None of the values in the sample are "numeric"
            pass

    def _check_correlations(self) -> None:
        """
        Analyzes data redundancy.

        Alerts:
            - HIGH_CORRELATION (WARNING): > 0.99. Collinearity/Duplicate data.
        """
        correlations = self.profile.get("correlations", {})

        for method in ["pearson", "spearman"]:
            correlation_matrix = correlations.get(method, [])
            if not correlation_matrix:
                continue

            checked_pairs = set()

            for row in correlation_matrix:
                primary_column = row.get("column")
                if not primary_column:
                    continue

                for comparison_column, value in row.items():
                    # SKIP 1: The 'column' key (it's the label, not a data point)
                    # SKIP 2: The Diagonal (Variable A vs Variable A is always 1.0)
                    if comparison_column in ("column", primary_column):
                        continue

                    # SKIP 3: Missing correlations (NaN/None)
                    if value is None:
                        continue

                    # CANONICALIZATION:
                    # Sort the pair so that ("Age", "Salary") and ("Salary", "Age")
                    # result in the exact same tuple ID: ('Age', 'Salary').
                    pair = tuple(sorted((primary_column, comparison_column)))
                    # If we have already seen this pair (from the other direction), skip it.
                    if pair in checked_pairs:
                        continue

                    if abs(value) > config.HIGH_CORRELATION_THRESHOLD:
                        self.alerts.append(
                            Alert(
                                column=f"{primary_column} <-> {comparison_column}",
                                alert_type="HIGH_CORRELATION",
                                level=AlertLevel.WARNING,
                                message=(
                                    f"Columns are highly correlated ({value:.4f}) via {method}. "
                                    "They contain redundant information."
                                ),
                                value=value,
                            )
                        )

                        # Mark this pair as "seen" so we don't report it again
                        checked_pairs.add(pair)
