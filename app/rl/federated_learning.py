"""
Federated Learning System for DataMCPServerAgent.
This module implements federated learning capabilities for distributed RL
training across multiple organizations while preserving privacy.
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from app.core.logging import get_logger
except ImportError:
    from app.core.simple_logging import get_logger

try:
    from app.monitoring.rl_analytics import get_metrics_collector
except ImportError:
    # Create a simple fallback metrics collector
    class SimpleMetricsCollector:
        def record_metric(self, _name, _value, _tags=None):
            pass

        def record_event(self, _name, _data, _level="info"):
            pass

    def get_metrics_collector():
        return SimpleMetricsCollector()

logger = get_logger(__name__)


class FederatedRole(str, Enum):
    """Federated learning roles."""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    AGGREGATOR = "aggregator"


class FederationStatus(str, Enum):
    """Federation status."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class PrivacyLevel(str, Enum):
    """Privacy protection levels."""
    NONE = "none"
    DIFFERENTIAL = "differential"
    HOMOMORPHIC = "homomorphic"
    SECURE_AGGREGATION = "secure_aggregation"


@dataclass
class FederatedParticipant:
    """Represents a participant in federated learning."""
    participant_id: str
    name: str
    organization: str
    endpoint: str
    public_key: Optional[str] = None
    data_size: int = 0
    last_seen: float = 0
    contribution_weight: float = 1.0
    privacy_budget: float = 1.0
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FederatedRound:
    """Represents a round of federated learning."""
    round_id: str
    round_number: int
    started_at: float
    completed_at: Optional[float] = None
    participants: List[str] = None
    global_model_hash: str = ""
    aggregated_metrics: Dict[str, float] = None
    privacy_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.participants is None:
            self.participants = []
        if self.aggregated_metrics is None:
            self.aggregated_metrics = {}
        if self.privacy_metrics is None:
            self.privacy_metrics = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DifferentialPrivacy:
    """Implements differential privacy mechanisms."""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """Initialize differential privacy.

        Args:
            epsilon: Privacy budget parameter
            delta: Probability of privacy breach
        """
        self.epsilon = epsilon
        self.delta = delta
        self.noise_scale = self._compute_noise_scale()

    def _compute_noise_scale(self) -> float:
        """Compute noise scale for Gaussian mechanism.

        Returns:
            Noise scale
        """
        # Simplified noise scale computation
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def add_noise(
        self, tensor: torch.Tensor, sensitivity: float = 1.0
    ) -> torch.Tensor:
        """Add differential privacy noise to tensor.

        Args:
            tensor: Input tensor
            sensitivity: Sensitivity of the function

        Returns:
            Noisy tensor
        """
        noise = torch.normal(
            mean=0.0,
            std=self.noise_scale * sensitivity,
            size=tensor.shape,
            device=tensor.device
        )

        return tensor + noise

    def clip_gradients(
        self, gradients: torch.Tensor, max_norm: float = 1.0
    ) -> torch.Tensor:
        """Clip gradients for privacy.

        Args:
            gradients: Input gradients
            max_norm: Maximum norm for clipping

        Returns:
            Clipped gradients
        """
        grad_norm = torch.norm(gradients)

        if grad_norm > max_norm:
            gradients = gradients * (max_norm / grad_norm)

        return gradients


class SecureAggregation:
    """Implements secure aggregation for federated learning."""

    def __init__(self, num_participants: int):
        """Initialize secure aggregation.

        Args:
            num_participants: Number of participants
        """
        self.num_participants = num_participants
        self.threshold = max(1, num_participants // 2)  # Majority threshold

    def generate_masks(self, model_size: int) -> Dict[str, torch.Tensor]:
        """Generate random masks for secure aggregation.

        Args:
            model_size: Size of model parameters

        Returns:
            Dictionary of masks for each participant
        """
        masks = {}

        for i in range(self.num_participants):
            participant_id = f"participant_{i}"
            mask = torch.randn(model_size) * 0.1  # Small random mask
            masks[participant_id] = mask

        return masks

    def aggregate_with_masks(
        self,
        masked_updates: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Aggregate masked updates.

        Args:
            masked_updates: Masked model updates from participants
            masks: Masks used by participants

        Returns:
            Aggregated update
        """
        if len(masked_updates) < self.threshold:
            raise ValueError(
                f"Insufficient participants: {len(masked_updates)} < "
                f"{self.threshold}"
            )

        # Sum masked updates
        total_masked = sum(masked_updates.values())

        # Sum masks from participating clients
        participating_masks = {
            pid: mask for pid, mask in masks.items()
            if pid in masked_updates
        }
        total_mask = sum(participating_masks.values())

        # Remove mask to get true aggregate
        aggregated = total_masked - total_mask

        return aggregated / len(masked_updates)


class FederatedLearningCoordinator:
    """Coordinates federated learning across multiple participants."""

    def __init__(
        self,
        federation_id: str,
        privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL,
        min_participants: int = 2,
        max_rounds: int = 100,
        convergence_threshold: float = 0.01
    ):
        """Initialize federated learning coordinator.

        Args:
            federation_id: Unique federation identifier
            privacy_level: Privacy protection level
            min_participants: Minimum participants required
            max_rounds: Maximum training rounds
            convergence_threshold: Convergence threshold
        """
        self.federation_id = federation_id
        self.privacy_level = privacy_level
        self.min_participants = min_participants
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold

        # Federation state
        self.participants: Dict[str, FederatedParticipant] = {}
        self.rounds: List[FederatedRound] = []
        self.global_model: Optional[nn.Module] = None
        self.status = FederationStatus.INITIALIZING

        # Privacy mechanisms
        self.differential_privacy = DifferentialPrivacy()
        self.secure_aggregation = None

        # Metrics
        self.metrics_collector = get_metrics_collector()
        self.federation_metrics = defaultdict(list)

        logger.info(
            f"🤝 Initialized federated learning coordinator: {federation_id}"
        )

    def register_participant(
        self,
        participant_id: str,
        name: str,
        organization: str,
        endpoint: str,
        data_size: int = 0,
        public_key: Optional[str] = None
    ) -> bool:
        """Register a new participant.

        Args:
            participant_id: Unique participant identifier
            name: Participant name
            organization: Organization name
            endpoint: Communication endpoint
            data_size: Size of local dataset
            public_key: Public key for encryption

        Returns:
            True if registration successful
        """
        if participant_id in self.participants:
            logger.warning(f"Participant {participant_id} already registered")
            return False

        participant = FederatedParticipant(
            participant_id=participant_id,
            name=name,
            organization=organization,
            endpoint=endpoint,
            data_size=data_size,
            public_key=public_key,
            last_seen=time.time(),
        )

        self.participants[participant_id] = participant

        # Update secure aggregation if needed
        if self.privacy_level == PrivacyLevel.SECURE_AGGREGATION:
            self.secure_aggregation = SecureAggregation(
                len(self.participants)
            )

        logger.info(f"📝 Registered participant: {name} ({organization})")

        # Record registration event
        self.metrics_collector.record_event(
            "federated_participant_registered",
            {
                "federation_id": self.federation_id,
                "participant_id": participant_id,
                "organization": organization,
                "data_size": data_size,
            },
            "info"
        )

        return True

    def start_federation(self, initial_model: nn.Module) -> bool:
        """Start the federated learning process.

        Args:
            initial_model: Initial global model

        Returns:
            True if started successfully
        """
        if len(self.participants) < self.min_participants:
            logger.error(
                f"Insufficient participants: {len(self.participants)} < "
                f"{self.min_participants}"
            )
            return False

        self.global_model = initial_model
        self.status = FederationStatus.ACTIVE

        logger.info(
            f"🚀 Started federated learning with {len(self.participants)} "
            f"participants"
        )

        # Record federation start
        self.metrics_collector.record_event(
            "federated_learning_started",
            {
                "federation_id": self.federation_id,
                "participants": len(self.participants),
                "privacy_level": self.privacy_level.value,
            },
            "info"
        )

        return True

    async def run_federated_round(self) -> Optional[FederatedRound]:
        """Run a single round of federated learning.

        Returns:
            Federated round results or None if failed
        """
        if self.status != FederationStatus.ACTIVE:
            logger.error("Federation not active")
            return None

        round_number = len(self.rounds) + 1
        round_id = f"{self.federation_id}_round_{round_number}"

        logger.info(f"🔄 Starting federated round {round_number}")

        # Create new round
        fed_round = FederatedRound(
            round_id=round_id,
            round_number=round_number,
            started_at=time.time(),
        )

        try:
            # Select participants for this round
            active_participants = self._select_participants()
            fed_round.participants = list(active_participants.keys())

            if len(active_participants) < self.min_participants:
                logger.warning(
                    f"Insufficient active participants: "
                    f"{len(active_participants)}"
                )
                return None

            # Distribute global model to participants
            await self._distribute_global_model(active_participants)

            # Collect local updates from participants
            local_updates = await self._collect_local_updates(
                active_participants
            )

            if not local_updates:
                logger.error("No local updates received")
                return None

            # Aggregate updates
            aggregated_update = await self._aggregate_updates(local_updates)

            # Update global model
            self._update_global_model(aggregated_update)

            # Compute round metrics
            fed_round.aggregated_metrics = await self._compute_round_metrics(
                local_updates
            )
            fed_round.privacy_metrics = self._compute_privacy_metrics()
            fed_round.global_model_hash = self._compute_model_hash()
            fed_round.completed_at = time.time()

            # Store round
            self.rounds.append(fed_round)

            # Record metrics
            self.metrics_collector.record_metric(
                "federated_round_duration",
                fed_round.completed_at - fed_round.started_at,
                {"federation_id": self.federation_id, "round": round_number}
            )

            logger.info(f"✅ Completed federated round {round_number}")

            return fed_round

        except Exception as e:
            logger.error(f"Error in federated round {round_number}: {e}")
            fed_round.completed_at = time.time()
            return fed_round

    def _select_participants(self) -> Dict[str, FederatedParticipant]:
        """Select participants for the current round.

        Returns:
            Dictionary of selected participants
        """
        # Simple selection: all active participants
        active_participants = {
            pid: participant for pid, participant in self.participants.items()
            if (participant.status == "active" and
                time.time() - participant.last_seen < 3600)
        }

        return active_participants

    async def _distribute_global_model(
        self, participants: Dict[str, FederatedParticipant]
    ):
        """Distribute global model to participants.

        Args:
            participants: Selected participants
        """
        # Simulate model distribution
        logger.info(
            f"📤 Distributing global model to {len(participants)} participants"
        )

        for _participant_id, participant in participants.items():
            # In real implementation, this would send the model over network
            logger.debug(f"Sent model to {participant.name}")
            participant.last_seen = time.time()

        await asyncio.sleep(1)  # Simulate network delay

    async def _collect_local_updates(
        self,
        participants: Dict[str, FederatedParticipant]
    ) -> Dict[str, torch.Tensor]:
        """Collect local model updates from participants.

        Args:
            participants: Selected participants

        Returns:
            Dictionary of local updates
        """
        logger.info(
            f"📥 Collecting updates from {len(participants)} participants"
        )

        local_updates = {}

        for participant_id, participant in participants.items():
            # Simulate local training and update generation
            await asyncio.sleep(0.5)  # Simulate training time

            # Generate mock update (in real implementation, this comes from
            # participant)
            if self.global_model:
                update = (
                    torch.randn_like(next(self.global_model.parameters())) *
                    0.01
                )

                # Apply privacy protection
                if self.privacy_level == PrivacyLevel.DIFFERENTIAL:
                    update = self.differential_privacy.add_noise(update)
                    update = self.differential_privacy.clip_gradients(update)

                local_updates[participant_id] = update

                logger.debug(f"Received update from {participant.name}")

        return local_updates

    async def _aggregate_updates(
        self, local_updates: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Aggregate local updates into global update.

        Args:
            local_updates: Local updates from participants

        Returns:
            Aggregated global update
        """
        logger.info(f"🔄 Aggregating {len(local_updates)} local updates")

        if (self.privacy_level == PrivacyLevel.SECURE_AGGREGATION and
                self.secure_aggregation):
            # Use secure aggregation
            model_size = next(iter(local_updates.values())).numel()
            masks = self.secure_aggregation.generate_masks(model_size)

            # Apply masks to updates
            masked_updates = {}
            for participant_id, update in local_updates.items():
                if participant_id in masks:
                    masked_updates[participant_id] = (
                        update + masks[participant_id]
                    )

            aggregated = self.secure_aggregation.aggregate_with_masks(
                masked_updates, masks
            )
        else:
            # Simple federated averaging
            weights = []
            updates = []

            for participant_id, update in local_updates.items():
                participant = self.participants[participant_id]
                weight = (
                    participant.contribution_weight * participant.data_size
                )
                weights.append(weight)
                updates.append(update)

            # Weighted average
            total_weight = sum(weights)
            if total_weight > 0:
                aggregated = (
                    sum(w * u for w, u in zip(weights, updates)) /
                    total_weight
                )
            else:
                aggregated = sum(updates) / len(updates)

        return aggregated

    def _update_global_model(self, aggregated_update: torch.Tensor):
        """Update global model with aggregated update.

        Args:
            aggregated_update: Aggregated update from participants
        """
        if self.global_model is None:
            return

        # Apply update to first parameter (simplified)
        with torch.no_grad():
            param = next(self.global_model.parameters())
            param.data += aggregated_update.view_as(param.data)

        logger.debug("Updated global model with aggregated update")

    async def _compute_round_metrics(
        self, local_updates: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute metrics for the current round.

        Args:
            local_updates: Local updates from participants

        Returns:
            Round metrics
        """
        metrics = {}

        # Participation rate
        metrics["participation_rate"] = (
            len(local_updates) / len(self.participants)
        )

        # Update diversity (variance of updates)
        if len(local_updates) > 1:
            updates_tensor = torch.stack(list(local_updates.values()))
            metrics["update_variance"] = torch.var(updates_tensor).item()
        else:
            metrics["update_variance"] = 0.0

        # Communication efficiency (mock)
        metrics["communication_rounds"] = len(self.rounds) + 1

        return metrics

    def _compute_privacy_metrics(self) -> Dict[str, float]:
        """Compute privacy metrics.

        Returns:
            Privacy metrics
        """
        metrics = {}

        if self.privacy_level == PrivacyLevel.DIFFERENTIAL:
            metrics["epsilon_spent"] = self.differential_privacy.epsilon
            metrics["delta"] = self.differential_privacy.delta
            metrics["noise_scale"] = self.differential_privacy.noise_scale

        # Mock privacy score
        metrics["privacy_level"] = hash(self.privacy_level.value) % 100

        return metrics

    def _compute_model_hash(self) -> str:
        """Compute hash of global model.

        Returns:
            Model hash
        """
        if self.global_model is None:
            return ""

        # Simple hash of model parameters
        param_str = ""
        for param in self.global_model.parameters():
            param_str += str(param.data.sum().item())

        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    def get_federation_status(self) -> Dict[str, Any]:
        """Get federation status and metrics.

        Returns:
            Federation status
        """
        return {
            "federation_id": self.federation_id,
            "status": self.status.value,
            "participants": len(self.participants),
            "rounds_completed": len(self.rounds),
            "privacy_level": self.privacy_level.value,
            "last_round": (
                self.rounds[-1].to_dict() if self.rounds else None
            ),
            "participant_details": [
                p.to_dict() for p in self.participants.values()
            ],
        }

    async def stop_federation(self):
        """Stop the federated learning process."""
        self.status = FederationStatus.COMPLETED

        logger.info(f"🛑 Stopped federated learning: {self.federation_id}")

        # Record federation completion
        self.metrics_collector.record_event(
            "federated_learning_completed",
            {
                "federation_id": self.federation_id,
                "total_rounds": len(self.rounds),
                "participants": len(self.participants),
            },
            "info"
        )


# Global federated learning coordinators
_federated_coordinators: Dict[str, FederatedLearningCoordinator] = {}


def create_federated_coordinator(
    federation_id: str,
    privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL,
    min_participants: int = 2,
    max_rounds: int = 100
) -> FederatedLearningCoordinator:
    """Create a new federated learning coordinator.

    Args:
        federation_id: Unique federation identifier
        privacy_level: Privacy protection level
        min_participants: Minimum participants required
        max_rounds: Maximum training rounds

    Returns:
        Federated learning coordinator
    """
    global _federated_coordinators

    if federation_id in _federated_coordinators:
        return _federated_coordinators[federation_id]

    coordinator = FederatedLearningCoordinator(
        federation_id=federation_id,
        privacy_level=privacy_level,
        min_participants=min_participants,
        max_rounds=max_rounds,
    )

    _federated_coordinators[federation_id] = coordinator

    return coordinator


def get_federated_coordinator(
    federation_id: str
) -> Optional[FederatedLearningCoordinator]:
    """Get existing federated learning coordinator.

    Args:
        federation_id: Federation identifier

    Returns:
        Federated learning coordinator or None
    """
    return _federated_coordinators.get(federation_id)
