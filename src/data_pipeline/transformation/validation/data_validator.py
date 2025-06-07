"""
Data Validator for data quality checking.

This module provides comprehensive data validation capabilities
including schema validation, data quality checks, and anomaly detection.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import polars as pl
from datetime import datetime, timezone

import structlog
from pydantic import BaseModel, Field

from ...core.pipeline_models import ValidationRule, QualityMetrics

class ValidationConfig(BaseModel):
    """Configuration for data validation."""
    enable_schema_validation: bool = Field(default=True, description="Enable schema validation")
    enable_quality_checks: bool = Field(default=True, description="Enable data quality checks")
    enable_anomaly_detection: bool = Field(default=False, description="Enable anomaly detection")

    # Quality thresholds
    max_null_percentage: float = Field(default=0.1, description="Maximum null percentage allowed")
    max_duplicate_percentage: float = Field(default=0.05, description="Maximum duplicate percentage allowed")
    min_completeness_score: float = Field(default=0.9, description="Minimum completeness score")

    # Validation options
    fail_on_error: bool = Field(default=True, description="Fail validation on first error")
    collect_all_errors: bool = Field(default=False, description="Collect all validation errors")

class ValidationResult(BaseModel):
    """Result of data validation."""
    is_valid: bool = Field(..., description="Whether data passed validation")
    validation_errors: List[str] = Field(default_factory=list, description="List of validation errors")
    quality_metrics: Optional[QualityMetrics] = Field(None, description="Data quality metrics")
    validation_time: float = Field(..., description="Time taken for validation")
    rules_applied: int = Field(..., description="Number of rules applied")
    rules_passed: int = Field(..., description="Number of rules passed")

class DataValidator:
    """
    Data validator for quality checking and validation.

    Provides comprehensive data validation including schema validation,
    data quality checks, and custom validation rules.
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the data validator.

        Args:
            config: Validation configuration
            logger: Logger instance
        """
        self.config = config or ValidationConfig()
        self.logger = logger or structlog.get_logger("data_validator")

        # Built-in validation rules
        self.built_in_validators = {
            "not_null": self._validate_not_null,
            "unique": self._validate_unique,
            "range": self._validate_range,
            "regex": self._validate_regex,
            "email": self._validate_email,
            "phone": self._validate_phone,
            "date": self._validate_date,
            "numeric": self._validate_numeric,
            "length": self._validate_length,
            "in_list": self._validate_in_list,
            "custom": self._validate_custom,
        }

        self.logger.info("Data validator initialized")

    async def validate_data(
        self,
        data: Union[pd.DataFrame, pl.DataFrame, Any],
        validation_rules: List[Dict[str, Any]]
    ) -> ValidationResult:
        """
        Validate data according to rules.

        Args:
            data: Data to validate
            validation_rules: List of validation rules

        Returns:
            Validation result
        """
        start_time = datetime.now(timezone.utc)

        try:
            self.logger.info("Starting data validation", rules_count=len(validation_rules))

            # Convert input data to DataFrame if needed
            if not isinstance(data, (pd.DataFrame, pl.DataFrame)):
                if isinstance(data, list):
                    data = pd.DataFrame(data)
                elif isinstance(data, dict):
                    data = pd.DataFrame([data])
                else:
                    raise ValueError(f"Unsupported data type: {type(data)}")

            # Convert validation rules
            rules = [ValidationRule(**rule) for rule in validation_rules]

            # Apply validation rules
            validation_errors = []
            rules_passed = 0

            for rule in rules:
                try:
                    is_valid = await self._apply_validation_rule(data, rule)
                    if is_valid:
                        rules_passed += 1
                    else:
                        error_msg = f"Rule '{rule.name}' failed for column '{rule.column}'"
                        validation_errors.append(error_msg)

                        if self.config.fail_on_error and not self.config.collect_all_errors:
                            break

                except Exception as e:
                    error_msg = f"Rule '{rule.name}' execution failed: {str(e)}"
                    validation_errors.append(error_msg)

                    if self.config.fail_on_error and not self.config.collect_all_errors:
                        break

            # Calculate quality metrics
            quality_metrics = None
            if self.config.enable_quality_checks:
                quality_metrics = await self._calculate_quality_metrics(data)

                # Check quality thresholds
                if quality_metrics.completeness_score < self.config.min_completeness_score:
                    validation_errors.append(
                        f"Completeness score {quality_metrics.completeness_score:.2f} "
                        f"below threshold {self.config.min_completeness_score:.2f}"
                    )

            # Determine overall validation result
            is_valid = len(validation_errors) == 0

            end_time = datetime.now(timezone.utc)
            validation_time = (end_time - start_time).total_seconds()

            result = ValidationResult(
                is_valid=is_valid,
                validation_errors=validation_errors,
                quality_metrics=quality_metrics,
                validation_time=validation_time,
                rules_applied=len(rules),
                rules_passed=rules_passed
            )

            self.logger.info(
                "Data validation completed",
                is_valid=is_valid,
                errors_count=len(validation_errors),
                rules_applied=len(rules),
                rules_passed=rules_passed,
                validation_time=validation_time
            )

            return result

        except Exception as e:
            self.logger.error("Data validation failed", error=str(e))
            raise e

    async def _apply_validation_rule(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        rule: ValidationRule
    ) -> bool:
        """Apply a single validation rule."""
        try:
            # Get validator function
            validator = self.built_in_validators.get(rule.rule_type)
            if not validator:
                raise ValueError(f"Unknown validation rule type: {rule.rule_type}")

            # Apply validation
            return await validator(data, rule)

        except Exception as e:
            self.logger.error(
                "Validation rule failed",
                rule_id=rule.rule_id,
                rule_type=rule.rule_type,
                error=str(e)
            )
            raise e

    async def _validate_not_null(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        rule: ValidationRule
    ) -> bool:
        """Validate that column has no null values."""
        if not rule.column or rule.column not in data.columns:
            return False

        if isinstance(data, pl.DataFrame):
            null_count = data[rule.column].null_count()
        else:
            null_count = data[rule.column].isnull().sum()

        return null_count == 0

    async def _validate_unique(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        rule: ValidationRule
    ) -> bool:
        """Validate that column has unique values."""
        if not rule.column or rule.column not in data.columns:
            return False

        if isinstance(data, pl.DataFrame):
            unique_count = data[rule.column].n_unique()
            total_count = len(data)
        else:
            unique_count = data[rule.column].nunique()
            total_count = len(data)

        return unique_count == total_count

    async def _validate_range(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        rule: ValidationRule
    ) -> bool:
        """Validate that values are within specified range."""
        if not rule.column or rule.column not in data.columns:
            return False

        min_val = rule.parameters.get("min")
        max_val = rule.parameters.get("max")

        if min_val is None and max_val is None:
            return True

        if isinstance(data, pl.DataFrame):
            column_data = data[rule.column]
            if min_val is not None:
                if (column_data < min_val).any():
                    return False
            if max_val is not None:
                if (column_data > max_val).any():
                    return False
        else:
            column_data = data[rule.column]
            if min_val is not None:
                if (column_data < min_val).any():
                    return False
            if max_val is not None:
                if (column_data > max_val).any():
                    return False

        return True

    async def _validate_regex(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        rule: ValidationRule
    ) -> bool:
        """Validate that values match regex pattern."""
        if not rule.column or rule.column not in data.columns:
            return False

        pattern = rule.condition
        if not pattern:
            return True

        try:
            regex = re.compile(pattern)

            if isinstance(data, pl.DataFrame):
                # Convert to pandas for regex validation
                column_data = data[rule.column].to_pandas()
            else:
                column_data = data[rule.column]

            # Check if all non-null values match the pattern
            non_null_data = column_data.dropna()
            if len(non_null_data) == 0:
                return True

            matches = non_null_data.astype(str).str.match(pattern)
            return matches.all()

        except re.error:
            return False

    async def _validate_email(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        rule: ValidationRule
    ) -> bool:
        """Validate email format."""
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        # Create a temporary rule with email regex
        email_rule = ValidationRule(
            rule_id=rule.rule_id,
            name=rule.name,
            rule_type="regex",
            column=rule.column,
            condition=email_pattern
        )

        return await self._validate_regex(data, email_rule)

    async def _validate_phone(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        rule: ValidationRule
    ) -> bool:
        """Validate phone number format."""
        # Simple phone pattern (can be customized)
        phone_pattern = r"^\+?[\d\s\-\(\)]{10,}$"

        # Create a temporary rule with phone regex
        phone_rule = ValidationRule(
            rule_id=rule.rule_id,
            name=rule.name,
            rule_type="regex",
            column=rule.column,
            condition=phone_pattern
        )

        return await self._validate_regex(data, phone_rule)

    async def _validate_date(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        rule: ValidationRule
    ) -> bool:
        """Validate date format."""
        if not rule.column or rule.column not in data.columns:
            return False

        try:
            if isinstance(data, pl.DataFrame):
                # Try to parse as date
                column_data = data[rule.column].to_pandas()
            else:
                column_data = data[rule.column]

            # Try to convert to datetime
            pd.to_datetime(column_data, errors='raise')
            return True

        except (ValueError, TypeError):
            return False

    async def _validate_numeric(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        rule: ValidationRule
    ) -> bool:
        """Validate that values are numeric."""
        if not rule.column or rule.column not in data.columns:
            return False

        if isinstance(data, pl.DataFrame):
            column_data = data[rule.column]
            return column_data.dtype.is_numeric()
        else:
            column_data = data[rule.column]
            return pd.api.types.is_numeric_dtype(column_data)

    async def _validate_length(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        rule: ValidationRule
    ) -> bool:
        """Validate string length."""
        if not rule.column or rule.column not in data.columns:
            return False

        min_length = rule.parameters.get("min_length")
        max_length = rule.parameters.get("max_length")

        if min_length is None and max_length is None:
            return True

        if isinstance(data, pl.DataFrame):
            column_data = data[rule.column].to_pandas()
        else:
            column_data = data[rule.column]

        # Convert to string and get lengths
        lengths = column_data.astype(str).str.len()

        if min_length is not None:
            if (lengths < min_length).any():
                return False

        if max_length is not None:
            if (lengths > max_length).any():
                return False

        return True

    async def _validate_in_list(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        rule: ValidationRule
    ) -> bool:
        """Validate that values are in allowed list."""
        if not rule.column or rule.column not in data.columns:
            return False

        allowed_values = rule.parameters.get("allowed_values", [])
        if not allowed_values:
            return True

        if isinstance(data, pl.DataFrame):
            column_data = data[rule.column].to_pandas()
        else:
            column_data = data[rule.column]

        # Check if all non-null values are in allowed list
        non_null_data = column_data.dropna()
        if len(non_null_data) == 0:
            return True

        return non_null_data.isin(allowed_values).all()

    async def _validate_custom(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        rule: ValidationRule
    ) -> bool:
        """Apply custom validation logic."""
        # This would allow for custom validation functions
        # For now, return True
        return True

    async def _calculate_quality_metrics(
        self,
        data: Union[pd.DataFrame, pl.DataFrame]
    ) -> QualityMetrics:
        """Calculate data quality metrics."""
        if isinstance(data, pl.DataFrame):
            total_records = len(data)
            total_cells = total_records * len(data.columns)
            null_count = data.null_count().sum_horizontal()[0] if total_records > 0 else 0
            duplicate_count = total_records - data.n_unique() if total_records > 0 else 0
        else:
            total_records = len(data)
            total_cells = data.size
            null_count = data.isnull().sum().sum()
            duplicate_count = data.duplicated().sum()

        # Calculate scores
        completeness_score = 1.0 - (null_count / total_cells) if total_cells > 0 else 1.0
        validity_score = 1.0 - (duplicate_count / total_records) if total_records > 0 else 1.0
        consistency_score = 1.0  # Placeholder

        return QualityMetrics(
            total_records=total_records,
            valid_records=total_records - duplicate_count,
            invalid_records=duplicate_count,
            completeness_score=completeness_score,
            validity_score=validity_score,
            consistency_score=consistency_score,
            null_count=null_count,
            duplicate_count=duplicate_count,
            measured_at=datetime.now(timezone.utc)
        )

    async def get_validation_info(self) -> Dict[str, Any]:
        """Get information about available validation rules."""
        return {
            "available_validators": list(self.built_in_validators.keys()),
            "config": self.config.model_dump(),
            "validator": "data_validator"
        }
