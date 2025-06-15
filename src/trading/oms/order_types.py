"""
Advanced order types for institutional trading.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..core.base_models import BaseOrder
from ..core.enums import ExecutionAlgorithm, OrderSide, OrderType


@dataclass
class AlgorithmicOrder(BaseOrder):
    """Base class for algorithmic orders."""

    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.TWAP
    algorithm_parameters: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    participation_rate: Optional[float] = None  # For POV algorithms

    # Execution tracking
    child_orders: List[str] = field(default_factory=list)
    execution_progress: float = 0.0
    slices_completed: int = 0
    total_slices: int = 0

    def __post_init__(self):
        """Initialize algorithmic order parameters."""
        if self.start_time is None:
            self.start_time = datetime.utcnow()

        if self.end_time is None and self.algorithm in [
            ExecutionAlgorithm.TWAP,
            ExecutionAlgorithm.VWAP,
        ]:
            # Default to 1 hour execution window
            self.end_time = self.start_time + timedelta(hours=1)


@dataclass
class TWAPOrder(AlgorithmicOrder):
    """Time Weighted Average Price order."""

    algorithm: ExecutionAlgorithm = field(default=ExecutionAlgorithm.TWAP, init=False)
    slice_interval: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    randomize_timing: bool = True
    randomization_factor: float = 0.1  # 10% randomization

    def __post_init__(self):
        super().__post_init__()

        # Calculate total slices
        if self.end_time and self.start_time:
            total_duration = self.end_time - self.start_time
            self.total_slices = max(1, int(total_duration / self.slice_interval))

        # Set default parameters
        self.algorithm_parameters.update(
            {
                "slice_interval_seconds": self.slice_interval.total_seconds(),
                "randomize_timing": self.randomize_timing,
                "randomization_factor": self.randomization_factor,
            }
        )


@dataclass
class VWAPOrder(AlgorithmicOrder):
    """Volume Weighted Average Price order."""

    algorithm: ExecutionAlgorithm = field(default=ExecutionAlgorithm.VWAP, init=False)
    volume_profile: Optional[List[float]] = None  # Historical volume profile
    max_participation_rate: float = 0.2  # Maximum 20% of market volume
    min_participation_rate: float = 0.05  # Minimum 5% of market volume

    def __post_init__(self):
        super().__post_init__()

        # Set default parameters
        self.algorithm_parameters.update(
            {
                "max_participation_rate": self.max_participation_rate,
                "min_participation_rate": self.min_participation_rate,
                "volume_profile": self.volume_profile or [],
            }
        )


@dataclass
class ImplementationShortfallOrder(AlgorithmicOrder):
    """Implementation Shortfall algorithm order."""

    algorithm: ExecutionAlgorithm = field(
        default=ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL, init=False
    )
    risk_aversion: float = 0.5  # Risk aversion parameter (0-1)
    market_impact_model: Optional[str] = None
    volatility_estimate: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()

        # Set default parameters
        self.algorithm_parameters.update(
            {
                "risk_aversion": self.risk_aversion,
                "market_impact_model": self.market_impact_model or "linear",
                "volatility_estimate": self.volatility_estimate,
            }
        )


@dataclass
class IcebergOrder(BaseOrder):
    """Iceberg order that shows only a small portion of the total quantity."""

    order_type: OrderType = field(default=OrderType.ICEBERG, init=False)
    display_quantity: Decimal = Decimal("0")
    hidden_quantity: Decimal = Decimal("0")
    refresh_threshold: float = 0.1  # Refresh when 10% of display qty remains

    # Iceberg execution tracking
    current_slice: int = 0
    total_slices: int = 0
    slice_orders: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize iceberg parameters."""
        if self.display_quantity == 0:
            # Default to 10% of total quantity
            self.display_quantity = self.quantity * Decimal("0.1")

        self.hidden_quantity = self.quantity - self.display_quantity

        if self.display_quantity > 0:
            self.total_slices = int(self.quantity / self.display_quantity) + 1

    @property
    def remaining_hidden_quantity(self) -> Decimal:
        """Calculate remaining hidden quantity."""
        return self.quantity - self.filled_quantity - self.display_quantity


@dataclass
class PercentOfVolumeOrder(AlgorithmicOrder):
    """Percent of Volume (POV) algorithm order."""

    algorithm: ExecutionAlgorithm = field(default=ExecutionAlgorithm.PERCENT_OF_VOLUME, init=False)
    target_participation_rate: float = 0.1  # Target 10% of market volume
    max_participation_rate: float = 0.25  # Maximum 25% of market volume
    min_order_size: Decimal = Decimal("1")
    max_order_size: Optional[Decimal] = None

    def __post_init__(self):
        super().__post_init__()

        # Set default parameters
        self.algorithm_parameters.update(
            {
                "target_participation_rate": self.target_participation_rate,
                "max_participation_rate": self.max_participation_rate,
                "min_order_size": float(self.min_order_size),
                "max_order_size": float(self.max_order_size) if self.max_order_size else None,
            }
        )


@dataclass
class ArrivalPriceOrder(AlgorithmicOrder):
    """Arrival Price algorithm order."""

    algorithm: ExecutionAlgorithm = field(default=ExecutionAlgorithm.ARRIVAL_PRICE, init=False)
    urgency: float = 0.5  # Urgency parameter (0-1, higher = more aggressive)
    max_price_deviation: Optional[float] = None  # Maximum price deviation from arrival price
    arrival_price: Optional[Decimal] = None

    def __post_init__(self):
        super().__post_init__()

        # Set arrival price to current market price if not specified
        if self.arrival_price is None:
            self.arrival_price = self.price

        # Set default parameters
        self.algorithm_parameters.update(
            {
                "urgency": self.urgency,
                "max_price_deviation": self.max_price_deviation,
                "arrival_price": float(self.arrival_price) if self.arrival_price else None,
            }
        )


@dataclass
class ConditionalOrder(BaseOrder):
    """Conditional order that triggers based on market conditions."""

    trigger_condition: str = ""  # Condition expression
    trigger_price: Optional[Decimal] = None
    trigger_symbol: Optional[str] = None  # Different symbol for trigger
    is_triggered: bool = False
    trigger_time: Optional[datetime] = None

    # Condition types
    PRICE_ABOVE = "PRICE_ABOVE"
    PRICE_BELOW = "PRICE_BELOW"
    VOLUME_ABOVE = "VOLUME_ABOVE"
    TIME_BASED = "TIME_BASED"
    TECHNICAL_INDICATOR = "TECHNICAL_INDICATOR"


@dataclass
class BracketOrder(BaseOrder):
    """Bracket order with profit target and stop loss."""

    parent_order_id: Optional[str] = None
    profit_target_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None

    # Child order IDs
    profit_target_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None

    # Bracket parameters
    trailing_stop: bool = False
    trailing_amount: Optional[Decimal] = None


@dataclass
class MultiLegOrder(BaseOrder):
    """Multi-leg order for complex strategies."""

    legs: List[Dict[str, Any]] = field(default_factory=list)
    strategy_type: str = ""  # e.g., "SPREAD", "STRADDLE", "BUTTERFLY"
    net_price: Optional[Decimal] = None

    # Execution parameters
    all_or_none: bool = False
    leg_fill_ratio: Optional[Dict[str, float]] = None


# Order factory functions
def create_twap_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    duration_hours: float = 1.0,
    slice_interval_minutes: int = 5,
    **kwargs,
) -> TWAPOrder:
    """Create a TWAP order with specified parameters."""

    start_time = datetime.utcnow()
    end_time = start_time + timedelta(hours=duration_hours)
    slice_interval = timedelta(minutes=slice_interval_minutes)

    return TWAPOrder(
        symbol=symbol,
        side=side,
        quantity=quantity,
        start_time=start_time,
        end_time=end_time,
        slice_interval=slice_interval,
        **kwargs,
    )


def create_vwap_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    duration_hours: float = 1.0,
    max_participation: float = 0.2,
    **kwargs,
) -> VWAPOrder:
    """Create a VWAP order with specified parameters."""

    start_time = datetime.utcnow()
    end_time = start_time + timedelta(hours=duration_hours)

    return VWAPOrder(
        symbol=symbol,
        side=side,
        quantity=quantity,
        start_time=start_time,
        end_time=end_time,
        max_participation_rate=max_participation,
        **kwargs,
    )


def create_iceberg_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    display_percentage: float = 0.1,
    **kwargs,
) -> IcebergOrder:
    """Create an iceberg order with specified parameters."""

    display_quantity = quantity * Decimal(str(display_percentage))

    return IcebergOrder(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        display_quantity=display_quantity,
        **kwargs,
    )
