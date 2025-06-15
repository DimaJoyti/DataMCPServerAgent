"""
Order validation system for institutional trading.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

from ..core.base_models import BaseOrder
from ..core.enums import ASSET_CLASS_CONFIGS, EXCHANGE_CONFIGS, AssetClass, OrderSide, OrderType


@dataclass
class ValidationResult:
    """Result of order validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)


class OrderValidator:
    """Comprehensive order validation system."""

    def __init__(self):
        self.min_order_value = Decimal("1.00")
        self.max_order_value = Decimal("100000000.00")  # $100M
        self.max_quantity = Decimal("1000000")
        self.symbol_pattern = re.compile(r"^[A-Z]{2,10}(/[A-Z]{2,10})?$")

        # Risk limits
        self.max_position_concentration = 0.25  # 25% max position
        self.max_daily_turnover = Decimal("1000000000")  # $1B daily

    async def validate_order(self, order: BaseOrder) -> ValidationResult:
        """
        Comprehensive order validation.

        Args:
            order: Order to validate

        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        # Basic field validation
        self._validate_basic_fields(order, result)

        # Symbol validation
        self._validate_symbol(order, result)

        # Quantity validation
        self._validate_quantity(order, result)

        # Price validation
        self._validate_price(order, result)

        # Order type validation
        self._validate_order_type(order, result)

        # Exchange validation
        self._validate_exchange(order, result)

        # Asset class validation
        self._validate_asset_class(order, result)

        # Risk validation
        await self._validate_risk_limits(order, result)

        # Business rules validation
        self._validate_business_rules(order, result)

        return result

    def _validate_basic_fields(self, order: BaseOrder, result: ValidationResult) -> None:
        """Validate basic required fields."""

        if not order.symbol:
            result.add_error("Symbol is required")

        if not order.side:
            result.add_error("Order side is required")

        if not order.order_type:
            result.add_error("Order type is required")

        if order.quantity <= 0:
            result.add_error("Quantity must be positive")

        if order.order_id and len(order.order_id) > 50:
            result.add_error("Order ID too long (max 50 characters)")

    def _validate_symbol(self, order: BaseOrder, result: ValidationResult) -> None:
        """Validate trading symbol format."""

        if not self.symbol_pattern.match(order.symbol):
            result.add_error(f"Invalid symbol format: {order.symbol}")

        # Check for common symbol issues
        if len(order.symbol) < 2:
            result.add_error("Symbol too short")

        if len(order.symbol) > 20:
            result.add_error("Symbol too long")

        # Warn about unusual symbols
        if "/" not in order.symbol and len(order.symbol) > 6:
            result.add_warning(f"Unusual symbol format: {order.symbol}")

    def _validate_quantity(self, order: BaseOrder, result: ValidationResult) -> None:
        """Validate order quantity."""

        if order.quantity <= 0:
            result.add_error("Quantity must be positive")

        if order.quantity > self.max_quantity:
            result.add_error(f"Quantity exceeds maximum: {self.max_quantity}")

        # Check for fractional shares based on asset class
        if order.symbol in EXCHANGE_CONFIGS:
            exchange_config = EXCHANGE_CONFIGS[order.exchange]
            min_order_size = exchange_config.get("min_order_size", 1)

            if order.quantity < Decimal(str(min_order_size)):
                result.add_error(f"Quantity below minimum: {min_order_size}")

        # Warn about very small quantities
        if order.quantity < Decimal("0.001"):
            result.add_warning("Very small quantity may have execution issues")

        # Warn about very large quantities
        if order.quantity > Decimal("100000"):
            result.add_warning("Large quantity may have market impact")

    def _validate_price(self, order: BaseOrder, result: ValidationResult) -> None:
        """Validate order price."""

        # Price required for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None:
                result.add_error("Price required for limit orders")
            elif order.price <= 0:
                result.add_error("Price must be positive")

        # Stop price required for stop orders
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None:
                result.add_error("Stop price required for stop orders")
            elif order.stop_price <= 0:
                result.add_error("Stop price must be positive")

        # Price reasonableness checks
        if order.price is not None:
            if order.price > Decimal("1000000"):
                result.add_warning("Very high price - please verify")

            if order.price < Decimal("0.0001"):
                result.add_warning("Very low price - please verify")

        # Stop price vs price validation
        if order.price is not None and order.stop_price is not None:
            if order.side == OrderSide.BUY:
                if order.stop_price <= order.price:
                    result.add_error(
                        "Stop price should be above limit price for buy stop-limit orders"
                    )
            else:
                if order.stop_price >= order.price:
                    result.add_error(
                        "Stop price should be below limit price for sell stop-limit orders"
                    )

    def _validate_order_type(self, order: BaseOrder, result: ValidationResult) -> None:
        """Validate order type compatibility."""

        # Check if order type is supported by exchange
        if order.exchange and order.exchange in EXCHANGE_CONFIGS:
            supported_types = EXCHANGE_CONFIGS[order.exchange].get("supported_order_types", [])
            if order.order_type not in supported_types:
                result.add_error(f"Order type {order.order_type} not supported by {order.exchange}")

        # Algorithmic order validation
        if order.order_type in [OrderType.TWAP, OrderType.VWAP, OrderType.IMPLEMENTATION_SHORTFALL]:
            if order.quantity < Decimal("1000"):
                result.add_warning(
                    "Small quantity for algorithmic order - consider direct execution"
                )

    def _validate_exchange(self, order: BaseOrder, result: ValidationResult) -> None:
        """Validate exchange compatibility."""

        if order.exchange:
            if order.exchange not in EXCHANGE_CONFIGS:
                result.add_error(f"Unsupported exchange: {order.exchange}")
            else:
                exchange_config = EXCHANGE_CONFIGS[order.exchange]

                # Check trading hours (simplified)
                trading_hours = exchange_config.get("trading_hours")
                if trading_hours and trading_hours != "24/7":
                    result.add_warning(f"Check trading hours for {order.exchange}: {trading_hours}")

                # Check minimum order size
                min_size = exchange_config.get("min_order_size", 1)
                if order.quantity < Decimal(str(min_size)):
                    result.add_error(f"Order size below minimum for {order.exchange}: {min_size}")

                # Check maximum order size
                max_size = exchange_config.get("max_order_size", 1000000)
                if order.quantity > Decimal(str(max_size)):
                    result.add_error(f"Order size above maximum for {order.exchange}: {max_size}")

    def _validate_asset_class(self, order: BaseOrder, result: ValidationResult) -> None:
        """Validate asset class specific rules."""

        # Determine asset class from symbol (simplified)
        asset_class = self._determine_asset_class(order.symbol)

        if asset_class in ASSET_CLASS_CONFIGS:
            config = ASSET_CLASS_CONFIGS[asset_class]

            # Check short selling rules
            if order.side in [OrderSide.SHORT, OrderSide.SELL] and not config.get(
                "short_selling_allowed", True
            ):
                result.add_error(f"Short selling not allowed for {asset_class}")

            # Check fractional shares
            if order.quantity != int(order.quantity) and not config.get("fractional_shares", False):
                result.add_error(f"Fractional shares not allowed for {asset_class}")

    def _determine_asset_class(self, symbol: str) -> AssetClass:
        """Determine asset class from symbol."""

        # Simplified asset class determination
        if "USD" in symbol or "EUR" in symbol or "GBP" in symbol:
            if "BTC" in symbol or "ETH" in symbol:
                return AssetClass.CRYPTOCURRENCY
            else:
                return AssetClass.CURRENCY
        elif symbol.endswith("=F"):  # Futures convention
            return AssetClass.DERIVATIVE
        else:
            return AssetClass.EQUITY

    async def _validate_risk_limits(self, order: BaseOrder, result: ValidationResult) -> None:
        """Validate risk limits."""

        # Calculate order value
        if order.price is not None:
            order_value = order.quantity * order.price

            if order_value < self.min_order_value:
                result.add_error(f"Order value below minimum: {self.min_order_value}")

            if order_value > self.max_order_value:
                result.add_error(f"Order value exceeds maximum: {self.max_order_value}")

        # Position concentration check (would need position data)
        # This is a placeholder for actual position checking
        if order.quantity > Decimal("10000"):
            result.add_warning("Large position - check concentration limits")

        # Daily turnover check (would need daily volume data)
        # This is a placeholder for actual turnover checking
        if order.price and order.quantity * order.price > Decimal("10000000"):
            result.add_warning("Large order value - check daily limits")

    def _validate_business_rules(self, order: BaseOrder, result: ValidationResult) -> None:
        """Validate business-specific rules."""

        # Time in force validation
        if order.time_in_force and order.order_type == OrderType.MARKET:
            if order.time_in_force.value not in ["IOC", "FOK"]:
                result.add_warning("Market orders typically use IOC or FOK time in force")

        # Strategy validation
        if order.strategy_id and len(order.strategy_id) > 50:
            result.add_error("Strategy ID too long")

        # Account validation
        if order.account_id and len(order.account_id) > 50:
            result.add_error("Account ID too long")

        # Portfolio validation
        if order.portfolio_id and len(order.portfolio_id) > 50:
            result.add_error("Portfolio ID too long")

        # Tag validation
        if order.tags:
            if len(order.tags) > 10:
                result.add_warning("Many tags may impact performance")

            for key, value in order.tags.items():
                if len(str(key)) > 50 or len(str(value)) > 100:
                    result.add_error("Tag key/value too long")

    def add_custom_validator(self, validator_func: callable) -> None:
        """Add custom validation function."""
        # This would allow adding custom validation rules
        pass

    def set_risk_limits(
        self,
        min_order_value: Optional[Decimal] = None,
        max_order_value: Optional[Decimal] = None,
        max_quantity: Optional[Decimal] = None,
        max_position_concentration: Optional[float] = None,
    ) -> None:
        """Update risk limits."""

        if min_order_value is not None:
            self.min_order_value = min_order_value

        if max_order_value is not None:
            self.max_order_value = max_order_value

        if max_quantity is not None:
            self.max_quantity = max_quantity

        if max_position_concentration is not None:
            self.max_position_concentration = max_position_concentration
