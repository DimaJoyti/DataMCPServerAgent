"""
A/B Testing System for Reinforcement Learning in DataMCPServerAgent.
This module implements automated A/B testing for RL algorithms and configurations.
"""

import hashlib
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.core.logging_improved import get_logger
from app.monitoring.rl_analytics import get_metrics_collector

logger = get_logger(__name__)


class ExperimentStatus(str, Enum):
    """Experiment status enumeration."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class StatisticalSignificance(str, Enum):
    """Statistical significance levels."""
    NOT_SIGNIFICANT = "not_significant"
    MARGINALLY_SIGNIFICANT = "marginally_significant"  # p < 0.1
    SIGNIFICANT = "significant"  # p < 0.05
    HIGHLY_SIGNIFICANT = "highly_significant"  # p < 0.01


@dataclass
class ExperimentVariant:
    """Represents a variant in an A/B test."""
    name: str
    description: str
    config: Dict[str, Any]
    traffic_allocation: float  # 0.0 to 1.0
    is_control: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentMetric:
    """Represents a metric to track in an experiment."""
    name: str
    description: str
    metric_type: str  # 'conversion', 'continuous', 'count'
    primary: bool = False
    higher_is_better: bool = True
    minimum_detectable_effect: float = 0.05  # 5% minimum effect

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentResult:
    """Represents the result of an A/B test."""
    variant_name: str
    metric_name: str
    sample_size: int
    mean: float
    std: float
    confidence_interval: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["confidence_interval"] = list(result["confidence_interval"])
        return result


@dataclass
class Experiment:
    """Represents an A/B test experiment."""
    id: str
    name: str
    description: str
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    status: ExperimentStatus
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    target_sample_size: int = 1000
    confidence_level: float = 0.95
    power: float = 0.8
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["variants"] = [v.to_dict() for v in self.variants]
        result["metrics"] = [m.to_dict() for m in self.metrics]
        result["status"] = self.status.value
        return result


class StatisticalAnalyzer:
    """Performs statistical analysis for A/B tests."""

    @staticmethod
    def calculate_sample_size(
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """Calculate required sample size for A/B test.
        
        Args:
            baseline_rate: Baseline conversion rate
            minimum_detectable_effect: Minimum effect to detect
            alpha: Type I error rate
            power: Statistical power
            
        Returns:
            Required sample size per variant
        """
        from scipy import stats

        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Effect size
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        # Pooled proportion
        p_pooled = (p1 + p2) / 2

        # Sample size calculation
        numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
                    z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p2 - p1) ** 2

        sample_size = int(np.ceil(numerator / denominator))

        return max(sample_size, 100)  # Minimum 100 samples

    @staticmethod
    def perform_t_test(
        control_data: List[float],
        treatment_data: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Perform t-test between control and treatment groups.
        
        Args:
            control_data: Control group data
            treatment_data: Treatment group data
            alpha: Significance level
            
        Returns:
            T-test results
        """
        from scipy import stats

        if len(control_data) < 2 or len(treatment_data) < 2:
            return {
                "error": "Insufficient data for t-test",
                "p_value": 1.0,
                "significant": False,
            }

        # Perform Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_data) - 1) * np.var(control_data, ddof=1) +
             (len(treatment_data) - 1) * np.var(treatment_data, ddof=1)) /
            (len(control_data) + len(treatment_data) - 2)
        )

        cohens_d = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std

        # Determine significance
        if p_value < 0.01:
            significance = StatisticalSignificance.HIGHLY_SIGNIFICANT
        elif p_value < 0.05:
            significance = StatisticalSignificance.SIGNIFICANT
        elif p_value < 0.1:
            significance = StatisticalSignificance.MARGINALLY_SIGNIFICANT
        else:
            significance = StatisticalSignificance.NOT_SIGNIFICANT

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
            "significance_level": significance.value,
            "cohens_d": cohens_d,
            "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large",
            "control_mean": np.mean(control_data),
            "treatment_mean": np.mean(treatment_data),
            "control_std": np.std(control_data, ddof=1),
            "treatment_std": np.std(treatment_data, ddof=1),
            "control_n": len(control_data),
            "treatment_n": len(treatment_data),
        }

    @staticmethod
    def calculate_confidence_interval(
        data: List[float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for data.
        
        Args:
            data: Data points
            confidence_level: Confidence level
            
        Returns:
            Confidence interval (lower, upper)
        """
        from scipy import stats

        if len(data) < 2:
            mean_val = np.mean(data) if data else 0
            return (mean_val, mean_val)

        mean_val = np.mean(data)
        std_err = stats.sem(data)

        # Calculate margin of error
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, len(data) - 1)
        margin_error = t_critical * std_err

        return (mean_val - margin_error, mean_val + margin_error)


class ABTestingEngine:
    """Main A/B testing engine for RL experiments."""

    def __init__(self):
        """Initialize A/B testing engine."""
        self.settings = get_settings()
        self.metrics_collector = get_metrics_collector()
        self.analyzer = StatisticalAnalyzer()

        # Experiment management
        self.experiments: Dict[str, Experiment] = {}
        self.experiment_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

        # Traffic allocation
        self.user_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)  # user_id -> experiment_id -> variant

        # Background tasks
        self.analysis_task = None
        self.is_running = False

    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[ExperimentVariant],
        metrics: List[ExperimentMetric],
        target_sample_size: int = 1000,
        confidence_level: float = 0.95
    ) -> str:
        """Create a new A/B test experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            variants: List of variants to test
            metrics: List of metrics to track
            target_sample_size: Target sample size
            confidence_level: Statistical confidence level
            
        Returns:
            Experiment ID
        """
        # Validate variants
        if len(variants) < 2:
            raise ValueError("At least 2 variants required")

        total_allocation = sum(v.traffic_allocation for v in variants)
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError("Traffic allocation must sum to 1.0")

        control_variants = [v for v in variants if v.is_control]
        if len(control_variants) != 1:
            raise ValueError("Exactly one control variant required")

        # Generate experiment ID
        experiment_id = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()[:8]

        # Create experiment
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            metrics=metrics,
            status=ExperimentStatus.DRAFT,
            target_sample_size=target_sample_size,
            confidence_level=confidence_level,
        )

        self.experiments[experiment_id] = experiment

        logger.info(f"📊 Created A/B test experiment: {name} (ID: {experiment_id})")

        return experiment_id

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B test experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if started successfully
        """
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.DRAFT:
            logger.error(f"Experiment {experiment_id} is not in draft status")
            return False

        # Start experiment
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = time.time()

        logger.info(f"🚀 Started A/B test experiment: {experiment.name}")

        # Record event
        self.metrics_collector.record_event(
            "ab_test_started",
            {
                "experiment_id": experiment_id,
                "experiment_name": experiment.name,
                "variants": len(experiment.variants),
                "metrics": len(experiment.metrics),
            },
            "info"
        )

        return True

    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an A/B test experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if stopped successfully
        """
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.RUNNING:
            logger.error(f"Experiment {experiment_id} is not running")
            return False

        # Stop experiment
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = time.time()

        logger.info(f"🛑 Stopped A/B test experiment: {experiment.name}")

        # Record event
        self.metrics_collector.record_event(
            "ab_test_stopped",
            {
                "experiment_id": experiment_id,
                "experiment_name": experiment.name,
                "duration": experiment.end_time - experiment.start_time,
            },
            "info"
        )

        return True

    def assign_user_to_variant(self, user_id: str, experiment_id: str) -> Optional[str]:
        """Assign user to a variant in an experiment.
        
        Args:
            user_id: User identifier
            experiment_id: Experiment ID
            
        Returns:
            Assigned variant name or None if experiment not found
        """
        if experiment_id not in self.experiments:
            return None

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.RUNNING:
            return None

        # Check if user already assigned
        if experiment_id in self.user_assignments[user_id]:
            return self.user_assignments[user_id][experiment_id]

        # Assign user to variant based on traffic allocation
        user_hash = int(hashlib.md5(f"{user_id}_{experiment_id}".encode()).hexdigest(), 16)
        random_value = (user_hash % 10000) / 10000.0  # 0.0 to 1.0

        cumulative_allocation = 0.0
        for variant in experiment.variants:
            cumulative_allocation += variant.traffic_allocation
            if random_value <= cumulative_allocation:
                self.user_assignments[user_id][experiment_id] = variant.name
                return variant.name

        # Fallback to control variant
        control_variant = next(v for v in experiment.variants if v.is_control)
        self.user_assignments[user_id][experiment_id] = control_variant.name
        return control_variant.name

    def record_metric(
        self,
        user_id: str,
        experiment_id: str,
        metric_name: str,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record a metric value for an experiment.
        
        Args:
            user_id: User identifier
            experiment_id: Experiment ID
            metric_name: Metric name
            value: Metric value
            context: Additional context
        """
        if experiment_id not in self.experiments:
            return

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.RUNNING:
            return

        # Get user's variant assignment
        variant_name = self.assign_user_to_variant(user_id, experiment_id)
        if not variant_name:
            return

        # Record metric data
        metric_data = {
            "user_id": user_id,
            "variant": variant_name,
            "metric": metric_name,
            "value": value,
            "timestamp": time.time(),
            "context": context or {},
        }

        self.experiment_data[experiment_id][variant_name].append(metric_data)

        # Record in metrics collector
        self.metrics_collector.record_metric(
            f"ab_test_{experiment_id}_{metric_name}",
            value,
            {
                "variant": variant_name,
                "experiment": experiment.name,
                "user_id": user_id,
            }
        )

    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Analysis results
        """
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}

        experiment = self.experiments[experiment_id]
        experiment_data = self.experiment_data[experiment_id]

        if not experiment_data:
            return {"error": "No data available for analysis"}

        # Get control variant
        control_variant = next(v for v in experiment.variants if v.is_control)
        control_data = experiment_data.get(control_variant.name, [])

        if not control_data:
            return {"error": "No control data available"}

        analysis_results = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "status": experiment.status.value,
            "start_time": experiment.start_time,
            "duration": (experiment.end_time or time.time()) - experiment.start_time if experiment.start_time else 0,
            "variants": {},
            "statistical_tests": {},
            "recommendations": [],
        }

        # Analyze each metric
        for metric in experiment.metrics:
            metric_name = metric.name

            # Get control data for this metric
            control_values = [
                d["value"] for d in control_data
                if d["metric"] == metric_name
            ]

            if not control_values:
                continue

            # Analyze each variant against control
            for variant in experiment.variants:
                if variant.is_control:
                    continue

                variant_data = experiment_data.get(variant.name, [])
                variant_values = [
                    d["value"] for d in variant_data
                    if d["metric"] == metric_name
                ]

                if not variant_values:
                    continue

                # Perform statistical test
                test_results = self.analyzer.perform_t_test(
                    control_values,
                    variant_values,
                    alpha=1 - experiment.confidence_level
                )

                # Calculate confidence intervals
                control_ci = self.analyzer.calculate_confidence_interval(
                    control_values, experiment.confidence_level
                )
                variant_ci = self.analyzer.calculate_confidence_interval(
                    variant_values, experiment.confidence_level
                )

                # Store results
                test_key = f"{variant.name}_vs_{control_variant.name}_{metric_name}"
                analysis_results["statistical_tests"][test_key] = {
                    "metric": metric_name,
                    "variant": variant.name,
                    "control": control_variant.name,
                    "test_results": test_results,
                    "control_ci": control_ci,
                    "variant_ci": variant_ci,
                    "sample_sizes": {
                        "control": len(control_values),
                        "variant": len(variant_values),
                    },
                }

                # Generate recommendations
                if test_results["significant"]:
                    improvement = (test_results["treatment_mean"] - test_results["control_mean"]) / test_results["control_mean"] * 100

                    if (improvement > 0 and metric.higher_is_better) or (improvement < 0 and not metric.higher_is_better):
                        analysis_results["recommendations"].append({
                            "type": "winner",
                            "variant": variant.name,
                            "metric": metric_name,
                            "improvement": abs(improvement),
                            "confidence": test_results["significance_level"],
                        })
                    else:
                        analysis_results["recommendations"].append({
                            "type": "loser",
                            "variant": variant.name,
                            "metric": metric_name,
                            "degradation": abs(improvement),
                            "confidence": test_results["significance_level"],
                        })

        return analysis_results

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment status and basic metrics.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment status
        """
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}

        experiment = self.experiments[experiment_id]
        experiment_data = self.experiment_data[experiment_id]

        # Calculate sample sizes
        variant_samples = {}
        for variant in experiment.variants:
            variant_data = experiment_data.get(variant.name, [])
            unique_users = len(set(d["user_id"] for d in variant_data))
            variant_samples[variant.name] = {
                "unique_users": unique_users,
                "total_events": len(variant_data),
            }

        total_users = sum(v["unique_users"] for v in variant_samples.values())
        progress = min(total_users / experiment.target_sample_size, 1.0) if experiment.target_sample_size > 0 else 0

        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "progress": progress,
            "total_users": total_users,
            "target_sample_size": experiment.target_sample_size,
            "variant_samples": variant_samples,
            "duration": (time.time() - experiment.start_time) if experiment.start_time else 0,
            "can_analyze": total_users >= 100,  # Minimum for analysis
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments.
        
        Returns:
            List of experiment summaries
        """
        experiments = []

        for experiment_id, experiment in self.experiments.items():
            status = self.get_experiment_status(experiment_id)
            experiments.append({
                "id": experiment_id,
                "name": experiment.name,
                "description": experiment.description,
                "status": experiment.status.value,
                "created_at": experiment.created_at,
                "start_time": experiment.start_time,
                "end_time": experiment.end_time,
                "variants": len(experiment.variants),
                "metrics": len(experiment.metrics),
                "progress": status.get("progress", 0),
                "total_users": status.get("total_users", 0),
            })

        return experiments


# Global A/B testing engine instance
_ab_testing_engine: Optional[ABTestingEngine] = None


def get_ab_testing_engine() -> ABTestingEngine:
    """Get global A/B testing engine."""
    global _ab_testing_engine
    if _ab_testing_engine is None:
        _ab_testing_engine = ABTestingEngine()
    return _ab_testing_engine
