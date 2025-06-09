"""
A/B Testing Service - Experimental framework for optimizing agent responses.
Enables controlled experiments to test different response strategies and personalities.
"""

import asyncio
import hashlib
import random
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app.core.logging import LoggerMixin, get_logger
from app.domain.models.base import BaseEntity, DomainService
from app.domain.models.brand_agent import BrandAgent, BrandPersonality

logger = get_logger(__name__)


class ExperimentStatus(str, Enum):
    """Status of an A/B test experiment."""
    
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ExperimentType(str, Enum):
    """Type of A/B test experiment."""
    
    PERSONALITY = "personality"
    RESPONSE_STRATEGY = "response_strategy"
    KNOWLEDGE_PRESENTATION = "knowledge_presentation"
    TONE_VARIATION = "tone_variation"
    LENGTH_VARIATION = "length_variation"
    CUSTOM = "custom"


class VariantAllocation(str, Enum):
    """How traffic is allocated to variants."""
    
    EQUAL = "equal"
    WEIGHTED = "weighted"
    GRADUAL_ROLLOUT = "gradual_rollout"


class ExperimentVariant:
    """A variant in an A/B test experiment."""
    
    def __init__(
        self,
        name: str,
        description: str,
        configuration: Dict[str, Any],
        traffic_percentage: float = 50.0,
        is_control: bool = False,
    ):
        self.id = str(uuid4())
        self.name = name
        self.description = description
        self.configuration = configuration
        self.traffic_percentage = traffic_percentage
        self.is_control = is_control
        
        # Metrics
        self.participant_count = 0
        self.conversion_count = 0
        self.total_satisfaction = 0.0
        self.total_response_time = 0.0
        self.escalation_count = 0
        self.resolution_count = 0
        
        self.created_at = datetime.now(timezone.utc)
    
    def add_result(
        self,
        satisfaction: Optional[float] = None,
        response_time_ms: Optional[float] = None,
        escalated: bool = False,
        resolved: bool = False,
    ) -> None:
        """Add a result to this variant."""
        self.participant_count += 1
        
        if satisfaction is not None:
            self.total_satisfaction += satisfaction
        
        if response_time_ms is not None:
            self.total_response_time += response_time_ms
        
        if escalated:
            self.escalation_count += 1
        
        if resolved:
            self.resolution_count += 1
            self.conversion_count += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get calculated metrics for this variant."""
        if self.participant_count == 0:
            return {
                "participants": 0,
                "avg_satisfaction": 0.0,
                "avg_response_time": 0.0,
                "escalation_rate": 0.0,
                "resolution_rate": 0.0,
                "conversion_rate": 0.0,
            }
        
        return {
            "participants": self.participant_count,
            "avg_satisfaction": self.total_satisfaction / self.participant_count,
            "avg_response_time": self.total_response_time / self.participant_count,
            "escalation_rate": self.escalation_count / self.participant_count,
            "resolution_rate": self.resolution_count / self.participant_count,
            "conversion_rate": self.conversion_count / self.participant_count,
        }


class ABTestExperiment(BaseEntity):
    """An A/B test experiment."""
    
    def __init__(
        self,
        name: str,
        description: str,
        experiment_type: ExperimentType,
        agent_id: str,
        variants: List[ExperimentVariant],
        allocation_method: VariantAllocation = VariantAllocation.EQUAL,
        target_sample_size: int = 1000,
        confidence_level: float = 0.95,
        minimum_effect_size: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.description = description
        self.experiment_type = experiment_type
        self.agent_id = agent_id
        self.variants = variants
        self.allocation_method = allocation_method
        self.target_sample_size = target_sample_size
        self.confidence_level = confidence_level
        self.minimum_effect_size = minimum_effect_size
        
        self.status = ExperimentStatus.DRAFT
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.winner_variant_id: Optional[str] = None
        
        # Ensure traffic percentages sum to 100%
        self._normalize_traffic_allocation()
    
    def _normalize_traffic_allocation(self) -> None:
        """Normalize traffic allocation to sum to 100%."""
        if self.allocation_method == VariantAllocation.EQUAL:
            percentage_per_variant = 100.0 / len(self.variants)
            for variant in self.variants:
                variant.traffic_percentage = percentage_per_variant
        else:
            total_percentage = sum(v.traffic_percentage for v in self.variants)
            if total_percentage != 100.0:
                for variant in self.variants:
                    variant.traffic_percentage = (variant.traffic_percentage / total_percentage) * 100.0
    
    def start_experiment(self) -> None:
        """Start the experiment."""
        self.status = ExperimentStatus.ACTIVE
        self.start_date = datetime.now(timezone.utc)
        self.version += 1
    
    def pause_experiment(self) -> None:
        """Pause the experiment."""
        self.status = ExperimentStatus.PAUSED
        self.version += 1
    
    def complete_experiment(self, winner_variant_id: Optional[str] = None) -> None:
        """Complete the experiment."""
        self.status = ExperimentStatus.COMPLETED
        self.end_date = datetime.now(timezone.utc)
        self.winner_variant_id = winner_variant_id
        self.version += 1
    
    def get_variant_for_user(self, user_id: str) -> ExperimentVariant:
        """Get the variant for a specific user (consistent assignment)."""
        # Use hash of user_id + experiment_id for consistent assignment
        hash_input = f"{user_id}:{self.id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        percentage = (hash_value % 10000) / 100.0  # 0-99.99%
        
        cumulative_percentage = 0.0
        for variant in self.variants:
            cumulative_percentage += variant.traffic_percentage
            if percentage < cumulative_percentage:
                return variant
        
        # Fallback to first variant
        return self.variants[0]
    
    def get_statistical_significance(self) -> Dict[str, Any]:
        """Calculate statistical significance of results."""
        if len(self.variants) != 2:
            return {"error": "Statistical significance calculation requires exactly 2 variants"}
        
        control_variant = next((v for v in self.variants if v.is_control), self.variants[0])
        test_variant = next((v for v in self.variants if not v.is_control), self.variants[1])
        
        control_metrics = control_variant.get_metrics()
        test_metrics = test_variant.get_metrics()
        
        # Simple statistical significance calculation (would use proper statistical tests in production)
        sample_size_adequate = (
            control_metrics["participants"] >= 100 and 
            test_metrics["participants"] >= 100
        )
        
        # Calculate effect size for conversion rate
        control_rate = control_metrics["conversion_rate"]
        test_rate = test_metrics["conversion_rate"]
        
        if control_rate > 0:
            relative_improvement = (test_rate - control_rate) / control_rate
        else:
            relative_improvement = 0.0
        
        # Mock p-value calculation (would use proper statistical test)
        if sample_size_adequate and abs(relative_improvement) > self.minimum_effect_size:
            p_value = 0.03  # Mock significant result
        else:
            p_value = 0.15  # Mock non-significant result
        
        is_significant = p_value < (1 - self.confidence_level)
        
        return {
            "is_significant": is_significant,
            "p_value": p_value,
            "confidence_level": self.confidence_level,
            "relative_improvement": relative_improvement,
            "effect_size": abs(relative_improvement),
            "sample_size_adequate": sample_size_adequate,
            "control_metrics": control_metrics,
            "test_metrics": test_metrics,
            "winner": "test" if is_significant and relative_improvement > 0 else "control" if is_significant else "inconclusive",
        }


class ABTestingService(DomainService, LoggerMixin):
    """Service for managing A/B testing experiments."""
    
    def __init__(self):
        super().__init__()
        self._experiments: Dict[str, ABTestExperiment] = {}
        self._active_experiments_by_agent: Dict[str, List[str]] = {}
    
    async def create_experiment(
        self,
        name: str,
        description: str,
        experiment_type: ExperimentType,
        agent_id: str,
        control_config: Dict[str, Any],
        test_configs: List[Dict[str, Any]],
        target_sample_size: int = 1000,
        confidence_level: float = 0.95,
        minimum_effect_size: float = 0.05,
    ) -> ABTestExperiment:
        """Create a new A/B test experiment."""
        
        # Create variants
        variants = []
        
        # Control variant
        control_variant = ExperimentVariant(
            name="Control",
            description="Original configuration",
            configuration=control_config,
            is_control=True,
        )
        variants.append(control_variant)
        
        # Test variants
        for i, test_config in enumerate(test_configs):
            test_variant = ExperimentVariant(
                name=f"Test {i + 1}",
                description=f"Test variant {i + 1}",
                configuration=test_config,
                is_control=False,
            )
            variants.append(test_variant)
        
        # Create experiment
        experiment = ABTestExperiment(
            name=name,
            description=description,
            experiment_type=experiment_type,
            agent_id=agent_id,
            variants=variants,
            target_sample_size=target_sample_size,
            confidence_level=confidence_level,
            minimum_effect_size=minimum_effect_size,
        )
        
        # Store experiment
        self._experiments[experiment.id] = experiment
        
        self.logger.info(f"Created experiment {experiment.id}: {name}")
        return experiment
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return False
        
        experiment.start_experiment()
        
        # Add to active experiments
        agent_id = experiment.agent_id
        if agent_id not in self._active_experiments_by_agent:
            self._active_experiments_by_agent[agent_id] = []
        self._active_experiments_by_agent[agent_id].append(experiment_id)
        
        self.logger.info(f"Started experiment {experiment_id}")
        return True
    
    async def get_variant_for_conversation(
        self, 
        agent_id: str, 
        user_id: str, 
        conversation_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get the appropriate variant configuration for a conversation."""
        
        # Get active experiments for this agent
        active_experiment_ids = self._active_experiments_by_agent.get(agent_id, [])
        
        for experiment_id in active_experiment_ids:
            experiment = self._experiments.get(experiment_id)
            if experiment and experiment.status == ExperimentStatus.ACTIVE:
                
                # Check if user should be included in experiment
                if await self._should_include_user(experiment, user_id, conversation_context):
                    variant = experiment.get_variant_for_user(user_id)
                    
                    return {
                        "experiment_id": experiment_id,
                        "variant_id": variant.id,
                        "variant_name": variant.name,
                        "configuration": variant.configuration,
                        "is_control": variant.is_control,
                    }
        
        return None
    
    async def record_experiment_result(
        self,
        experiment_id: str,
        variant_id: str,
        satisfaction: Optional[float] = None,
        response_time_ms: Optional[float] = None,
        escalated: bool = False,
        resolved: bool = False,
    ) -> None:
        """Record a result for an experiment variant."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return
        
        # Find the variant
        variant = next((v for v in experiment.variants if v.id == variant_id), None)
        if not variant:
            return
        
        # Record the result
        variant.add_result(satisfaction, response_time_ms, escalated, resolved)
        
        # Check if experiment should be completed
        total_participants = sum(v.participant_count for v in experiment.variants)
        if total_participants >= experiment.target_sample_size:
            await self._check_experiment_completion(experiment)
    
    async def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive results for an experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return None
        
        # Get variant metrics
        variant_results = []
        for variant in experiment.variants:
            metrics = variant.get_metrics()
            variant_results.append({
                "id": variant.id,
                "name": variant.name,
                "description": variant.description,
                "is_control": variant.is_control,
                "traffic_percentage": variant.traffic_percentage,
                "metrics": metrics,
            })
        
        # Get statistical analysis
        statistical_analysis = experiment.get_statistical_significance()
        
        # Calculate experiment duration
        duration_days = 0
        if experiment.start_date:
            end_date = experiment.end_date or datetime.now(timezone.utc)
            duration_days = (end_date - experiment.start_date).days
        
        return {
            "experiment": {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "type": experiment.experiment_type,
                "status": experiment.status,
                "start_date": experiment.start_date.isoformat() if experiment.start_date else None,
                "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
                "duration_days": duration_days,
                "target_sample_size": experiment.target_sample_size,
                "winner_variant_id": experiment.winner_variant_id,
            },
            "variants": variant_results,
            "statistical_analysis": statistical_analysis,
            "recommendations": await self._generate_recommendations(experiment, statistical_analysis),
        }
    
    async def create_personality_experiment(
        self,
        agent: BrandAgent,
        test_personality: BrandPersonality,
        experiment_name: str,
    ) -> ABTestExperiment:
        """Create an experiment to test a different personality."""
        
        control_config = {
            "personality": agent.personality.dict(),
            "type": "personality_test",
        }
        
        test_config = {
            "personality": test_personality.dict(),
            "type": "personality_test",
        }
        
        return await self.create_experiment(
            name=experiment_name,
            description=f"Testing personality variation for {agent.name}",
            experiment_type=ExperimentType.PERSONALITY,
            agent_id=agent.id,
            control_config=control_config,
            test_configs=[test_config],
        )
    
    async def create_response_strategy_experiment(
        self,
        agent_id: str,
        strategy_variations: List[Dict[str, Any]],
        experiment_name: str,
    ) -> ABTestExperiment:
        """Create an experiment to test different response strategies."""
        
        control_config = {
            "strategy": "default",
            "type": "response_strategy",
        }
        
        test_configs = [
            {
                "strategy": variation,
                "type": "response_strategy",
            }
            for variation in strategy_variations
        ]
        
        return await self.create_experiment(
            name=experiment_name,
            description="Testing different response strategies",
            experiment_type=ExperimentType.RESPONSE_STRATEGY,
            agent_id=agent_id,
            control_config=control_config,
            test_configs=test_configs,
        )
    
    async def _should_include_user(
        self, 
        experiment: ABTestExperiment, 
        user_id: str, 
        context: Dict[str, Any]
    ) -> bool:
        """Determine if a user should be included in an experiment."""
        
        # Basic inclusion criteria
        if not user_id:
            return False
        
        # Could add more sophisticated targeting criteria here
        # For example: user segment, conversation type, time of day, etc.
        
        return True
    
    async def _check_experiment_completion(self, experiment: ABTestExperiment) -> None:
        """Check if an experiment should be completed."""
        
        # Check sample size
        total_participants = sum(v.participant_count for v in experiment.variants)
        if total_participants < experiment.target_sample_size:
            return
        
        # Check statistical significance
        stats = experiment.get_statistical_significance()
        
        if stats.get("is_significant") and stats.get("sample_size_adequate"):
            # Determine winner
            winner = stats.get("winner")
            if winner == "test":
                winner_variant = next((v for v in experiment.variants if not v.is_control), None)
            elif winner == "control":
                winner_variant = next((v for v in experiment.variants if v.is_control), None)
            else:
                winner_variant = None
            
            # Complete experiment
            experiment.complete_experiment(winner_variant.id if winner_variant else None)
            
            # Remove from active experiments
            agent_id = experiment.agent_id
            if agent_id in self._active_experiments_by_agent:
                self._active_experiments_by_agent[agent_id].remove(experiment.id)
            
            self.logger.info(f"Completed experiment {experiment.id} with winner: {winner}")
    
    async def _generate_recommendations(
        self, 
        experiment: ABTestExperiment, 
        stats: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []
        
        if stats.get("is_significant"):
            winner = stats.get("winner")
            improvement = stats.get("relative_improvement", 0)
            
            if winner == "test":
                recommendations.append(f"Implement the test variant - it shows {improvement:.1%} improvement")
                recommendations.append("Monitor performance after rollout to confirm results")
            elif winner == "control":
                recommendations.append("Keep the current configuration - it performs better")
                recommendations.append("Consider testing other variations")
            
        else:
            recommendations.append("No significant difference found between variants")
            
            if not stats.get("sample_size_adequate"):
                recommendations.append("Consider running the experiment longer to gather more data")
            else:
                recommendations.append("The effect size may be too small to detect")
                recommendations.append("Consider testing more dramatic variations")
        
        return recommendations
