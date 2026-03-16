from dataclasses import dataclass
from enum import Enum

from netra_profiler.types import ColumnMetrics, NetraProfile, is_numeric


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
    POSSIBLE_NUMERIC_SAMPLE_SIZE: int = 5


# Initialize default config
config = DiagnosticConfig()


@dataclass
class Alert:
    column_name: str
    type: str  # e.g. "HIGH_NULLS"
    level: AlertLevel
    message: str
    value: float | None = None


class DiagnosticEngine:
    """
    Analyzes the NetraProfile object and generates a list of Alerts.
    This is a pure logic layer: it does not query the data.
    """

    def __init__(self, profile: NetraProfile):
        self.profile = profile
        self.dataset = self.profile.get("dataset", {})
        self.columns = self.profile.get("columns", {})
        self.correlations = self.profile.get("correlations", {})

        self.row_count = self.dataset.get("row_count", 0)
        self.alerts: list[Alert] = []

    def run(self) -> list[Alert]:
        """Runs all diagnostic checks and returns the alerts."""
        if self.row_count == 0:
            return []

        # 1. Scalar Checks (Column by Column)
        for column_name, column_profile in self.columns.items():
            self._check_nulls(column_name, column_profile)
            self._check_constant_and_id(column_name, column_profile)
            self._check_cardinality(column_name, column_profile)
            self._check_skew(column_name, column_profile)
            self._check_zeros(column_name, column_profile)
            self._check_possible_numeric(column_name, column_profile)

        # 2. Global/Relationship Checks
        self._check_correlations()

        return self.alerts

    def _check_nulls(self, column_name: str, column_profile: ColumnMetrics) -> None:
        """
        Analyzes missing data.

        Alerts:
            - EMPTY_COLUMN (CRITICAL): Nulls > 95%. Likely useless.
            - HIGH_NULLS (CRITICAL): Nulls > 50%. Hard to impute.
        """
        null_count = column_profile.get("null_count", 0)
        null_percentage = null_count / self.row_count

        if null_percentage > config.NULL_CRITICAL_THRESHOLD:
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="EMPTY_COLUMN",
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
                    column_name=column_name,
                    type="HIGH_NULLS",
                    level=AlertLevel.WARNING,
                    message=f"Column is {null_percentage:.1%} empty. Imputation may be difficult.",
                    value=null_percentage,
                )
            )

    def _check_constant_and_id(self, column_name: str, column_profile: ColumnMetrics) -> None:
        """
        Analyzes value variance and detects Primary Keys or Duplicate Keys.

        Alerts:
            - CONSTANT (CRITICAL): Only 1 unique value. Adds no information.
            - ALL_DISTINCT (CRITICAL): Unique count ~= Row count. Likely an ID/PII.
        """
        n_unique = column_profile.get("n_unique")
        column_data_type = column_profile.get("data_type", "")

        if not n_unique:
            return

        if n_unique == 1:
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="CONSTANT",
                    level=AlertLevel.CRITICAL,
                    message="Column has only 1 unique value. It adds no variance to the dataset.",
                    value=1.0,
                )
            )
            return

        # ID/PK Validation Logic
        # We strictly only evaluate Ints and Strings for PK status
        is_id_candidate = any(
            data_type in column_data_type for data_type in ["Int", "String", "Utf8", "Categorical"]
        )

        # Check for ALL_DISTINCT (ID columns)
        # We flag if the dataset is reasonably sized (> 100 rows)
        # and unique count is very close to row count (e.g. > 99%).
        if (
            is_id_candidate
            and self.row_count > config.MIN_ROWS_FOR_ID_CHECK
            and n_unique > (self.row_count * config.ID_UNIQUENESS_THRESHOLD)
        ):
            if n_unique == self.row_count:
                # Perfect Primary Key
                self.alerts.append(
                    Alert(
                        column_name=column_name,
                        type="ALL_DISTINCT",
                        level=AlertLevel.INFO,
                        message="Column is 100% distinct. Likely a Primary Key or ID.",
                        value=n_unique,
                    )
                )
            else:
                # Corrupted Primary Key
                duplicates = self.row_count - n_unique
                self.alerts.append(
                    Alert(
                        column_name=column_name,
                        type="DUPLICATE_KEYS",
                        level=AlertLevel.WARNING,
                        message=(
                            f"Column is almost unique, but contains {duplicates} duplicate values."
                        ),
                        value=n_unique,
                    )
                )

    def _check_cardinality(self, column_name: str, column_profile: ColumnMetrics) -> None:
        """
        Analyzes cardinality of string columns.

        Alerts:
            - HIGH_CARDINALITY (WARNING): > 10k unique values. Expensive for ML.
        """
        n_unique = column_profile.get("n_unique")
        data_type = column_profile.get("data_type", "")

        # Check if it is NOT a numeric type
        is_string_type = not is_numeric(data_type)

        # Only flag strings (Numeric high cardinality is normal)
        if (
            is_string_type
            and n_unique
            and n_unique > config.HIGH_CARDINALITY_THRESHOLD
            and n_unique < self.row_count
        ):
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="HIGH_CARDINALITY",
                    level=AlertLevel.WARNING,
                    message=f"High cardinality ({n_unique} unique values). Avoid One-Hot Encoding.",
                    value=n_unique,
                )
            )

    def _check_skew(self, column_name: str, column_profile: ColumnMetrics) -> None:
        """
        Analyzes distribution shape.

        Alerts:
            - SKEWED (WARNING): Skew > 2.0. Requires log-transformation.
        """
        skew = column_profile.get("skew")
        if skew is not None and abs(skew) > config.SKEW_THRESHOLD:
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="SKEWED",
                    level=AlertLevel.WARNING,
                    message=(
                        f"Distribution is highly skewed ({skew:.2f}). "
                        "Linear models may require transformation."
                    ),
                    value=skew,
                )
            )

    def _check_zeros(self, column_name: str, column_profile: ColumnMetrics) -> None:
        """
        Analyzes zero-inflation.

        Alerts:
            - ZERO_INFLATED (WARNING): > 10% zeros. Potential 'missing data' proxy.
        """
        data_type = column_profile.get("data_type", "")

        if not is_numeric(data_type):
            return

        zero_count = column_profile.get("zero_count", 0) or 0

        if zero_count > 0 and self.row_count > 0:
            zero_percentage = zero_count / self.row_count

            if zero_percentage > config.ZERO_INFLATED_THRESHOLD:
                self.alerts.append(
                    Alert(
                        column_name=column_name,
                        type="ZERO_INFLATED",
                        level=AlertLevel.WARNING,
                        message=(
                            f"Column is {zero_percentage:.1%} zeros. "
                            "Check if '0' represents missing data."
                        ),
                        value=zero_percentage,
                    )
                )

    def _check_possible_numeric(self, column_name: str, column_profile: ColumnMetrics) -> None:
        """
        Heuristic check for string columns that contain numbers.

        Alerts:
            - POSSIBLE_NUMERIC (INFO): Recommendation to cast type.
        """
        data_type = column_profile.get("data_type", "")

        # 1. Skip if the engine already knows it's a number
        if is_numeric(data_type):
            return

        # We check the Top-K most frequent values. If the top 5 values
        # can all be cast to float, it is highly likely the column is numeric.
        # We use Top-K because checking every row is expensive (O(N)),
        # while Top-K is O(1) here.
        top_k_list = column_profile.get("top_k", [])

        # 2. Check the sample (Top 5)
        # We only look at values that are NOT None
        sample_values = [
            top_k_item["value"]
            for top_k_item in top_k_list[: config.POSSIBLE_NUMERIC_SAMPLE_SIZE]
            if top_k_item["value"] is not None
        ]

        if not sample_values:
            return

        # 3. Try to convert all sampled values
        try:
            # If ANY value fails conversion, the whole column is treated as String
            # This is strict but safe.
            for value in sample_values:
                float(value)

            # If we survived the loop, they are all numbers
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="POSSIBLE_NUMERIC",
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
        # We explicitly extract the lists using string literals so mypy
        # statically knows these are list[CorrelationPair] and not the sampling_method string.
        pearson_matrix = self.correlations.get("pearson", [])
        spearman_matrix = self.correlations.get("spearman", [])

        for method, matrix in [("pearson", pearson_matrix), ("spearman", spearman_matrix)]:
            for edge in matrix:
                column_a = edge["column_a"]
                column_b = edge["column_b"]
                score = edge["score"]

                if abs(score) > config.HIGH_CORRELATION_THRESHOLD:
                    self.alerts.append(
                        Alert(
                            column_name=f"{column_a} <-> {column_b}",
                            type="HIGH_CORRELATION",
                            level=AlertLevel.WARNING,
                            message=(
                                f"Columns are highly correlated ({score:.4f}) via {method}. "
                                "They contain redundant information."
                            ),
                            value=score,
                        )
                    )
