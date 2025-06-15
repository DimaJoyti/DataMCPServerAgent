"""
Core utilities for the institutional trading system.
"""

import hashlib
import uuid
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Union


def generate_order_id(prefix: str = "ORD") -> str:
    """
    Generate a unique order ID.

    Args:
        prefix: Prefix for the order ID

    Returns:
        Unique order ID
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8].upper()
    return f"{prefix}_{timestamp}_{unique_id}"


def generate_trade_id(prefix: str = "TRD") -> str:
    """
    Generate a unique trade ID.

    Args:
        prefix: Prefix for the trade ID

    Returns:
        Unique trade ID
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8].upper()
    return f"{prefix}_{timestamp}_{unique_id}"


def calculate_position_size(
    portfolio_value: Decimal,
    risk_percentage: float,
    entry_price: Decimal,
    stop_loss_price: Optional[Decimal] = None,
) -> Decimal:
    """
    Calculate position size based on risk management rules.

    Args:
        portfolio_value: Total portfolio value
        risk_percentage: Risk percentage (0.01 = 1%)
        entry_price: Entry price per share
        stop_loss_price: Stop loss price (optional)

    Returns:
        Position size in shares
    """
    risk_amount = portfolio_value * Decimal(str(risk_percentage))

    if stop_loss_price is not None:
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share > 0:
            position_size = risk_amount / risk_per_share
        else:
            position_size = risk_amount / entry_price
    else:
        # Default to 2% risk per share if no stop loss
        position_size = risk_amount / (entry_price * Decimal("0.02"))

    return position_size.quantize(Decimal("1"), rounding=ROUND_HALF_UP)


def format_price(price: Union[Decimal, float], decimals: int = 2) -> str:
    """
    Format price for display.

    Args:
        price: Price to format
        decimals: Number of decimal places

    Returns:
        Formatted price string
    """
    if isinstance(price, float):
        price = Decimal(str(price))

    format_str = f"{{:,.{decimals}f}}"
    return format_str.format(float(price))


def format_quantity(quantity: Union[Decimal, float], decimals: int = 0) -> str:
    """
    Format quantity for display.

    Args:
        quantity: Quantity to format
        decimals: Number of decimal places

    Returns:
        Formatted quantity string
    """
    if isinstance(quantity, float):
        quantity = Decimal(str(quantity))

    format_str = f"{{:,.{decimals}f}}"
    return format_str.format(float(quantity))


def format_currency(amount: Union[Decimal, float], currency: str = "USD") -> str:
    """
    Format currency amount for display.

    Args:
        amount: Amount to format
        currency: Currency code

    Returns:
        Formatted currency string
    """
    if isinstance(amount, float):
        amount = Decimal(str(amount))

    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "CHF": "CHF ",
        "CAD": "C$",
        "AUD": "A$",
    }

    symbol = symbols.get(currency, f"{currency} ")
    return f"{symbol}{format_price(amount)}"


def calculate_percentage_change(old_value: Decimal, new_value: Decimal) -> float:
    """
    Calculate percentage change between two values.

    Args:
        old_value: Original value
        new_value: New value

    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0

    return float((new_value - old_value) / old_value * 100)


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> Optional[float]:
    """
    Calculate Sharpe ratio.

    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (annual)

    Returns:
        Sharpe ratio or None if insufficient data
    """
    if len(returns) < 2:
        return None

    import statistics

    excess_returns = [r - risk_free_rate / 252 for r in returns]  # Daily risk-free rate

    if statistics.stdev(excess_returns) == 0:
        return None

    return statistics.mean(excess_returns) / statistics.stdev(excess_returns) * (252**0.5)


def calculate_max_drawdown(values: List[float]) -> float:
    """
    Calculate maximum drawdown.

    Args:
        values: List of portfolio values

    Returns:
        Maximum drawdown percentage
    """
    if len(values) < 2:
        return 0.0

    peak = values[0]
    max_drawdown = 0.0

    for value in values[1:]:
        if value > peak:
            peak = value

        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown * 100


def calculate_volatility(returns: List[float], annualized: bool = True) -> Optional[float]:
    """
    Calculate volatility.

    Args:
        returns: List of returns
        annualized: Whether to annualize the volatility

    Returns:
        Volatility or None if insufficient data
    """
    if len(returns) < 2:
        return None

    import statistics

    volatility = statistics.stdev(returns)

    if annualized:
        volatility *= 252**0.5  # Annualize assuming 252 trading days

    return volatility


def calculate_var(
    returns: List[float], confidence_level: float = 0.95, portfolio_value: Optional[Decimal] = None
) -> Optional[float]:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: List of returns
        confidence_level: Confidence level (0.95 = 95%)
        portfolio_value: Portfolio value for absolute VaR

    Returns:
        VaR value
    """
    if len(returns) < 10:
        return None

    # Sort returns
    sorted_returns = sorted(returns)

    # Find percentile
    index = int((1 - confidence_level) * len(sorted_returns))
    var_return = sorted_returns[index]

    if portfolio_value is not None:
        return float(var_return * float(portfolio_value))

    return var_return


def calculate_beta(asset_returns: List[float], market_returns: List[float]) -> Optional[float]:
    """
    Calculate beta coefficient.

    Args:
        asset_returns: Asset returns
        market_returns: Market returns

    Returns:
        Beta coefficient or None if insufficient data
    """
    if len(asset_returns) != len(market_returns) or len(asset_returns) < 10:
        return None

    import statistics

    # Calculate covariance and variance
    asset_mean = statistics.mean(asset_returns)
    market_mean = statistics.mean(market_returns)

    covariance = sum(
        (a - asset_mean) * (m - market_mean) for a, m in zip(asset_returns, market_returns)
    ) / (len(asset_returns) - 1)

    market_variance = statistics.variance(market_returns)

    if market_variance == 0:
        return None

    return covariance / market_variance


def normalize_symbol(symbol: str) -> str:
    """
    Normalize trading symbol format.

    Args:
        symbol: Raw symbol

    Returns:
        Normalized symbol
    """
    # Remove spaces and convert to uppercase
    symbol = symbol.replace(" ", "").upper()

    # Handle common formats
    if ":" in symbol:
        # Exchange:Symbol format
        parts = symbol.split(":")
        if len(parts) == 2:
            symbol = parts[1]

    # Handle currency pairs
    if len(symbol) == 6 and "/" not in symbol:
        # EURUSD -> EUR/USD
        symbol = f"{symbol[:3]}/{symbol[3:]}"

    return symbol


def validate_price(price: Union[Decimal, float, str]) -> bool:
    """
    Validate price value.

    Args:
        price: Price to validate

    Returns:
        True if valid price
    """
    try:
        if isinstance(price, str):
            price = Decimal(price)
        elif isinstance(price, float):
            price = Decimal(str(price))

        return price > 0 and price < Decimal("1000000")
    except:
        return False


def validate_quantity(quantity: Union[Decimal, float, str]) -> bool:
    """
    Validate quantity value.

    Args:
        quantity: Quantity to validate

    Returns:
        True if valid quantity
    """
    try:
        if isinstance(quantity, str):
            quantity = Decimal(quantity)
        elif isinstance(quantity, float):
            quantity = Decimal(str(quantity))

        return quantity > 0 and quantity < Decimal("10000000")
    except:
        return False


def hash_order_data(order_data: Dict[str, Any]) -> str:
    """
    Generate hash for order data integrity.

    Args:
        order_data: Order data dictionary

    Returns:
        SHA256 hash
    """
    # Sort keys for consistent hashing
    sorted_data = {k: order_data[k] for k in sorted(order_data.keys())}

    # Convert to string
    data_str = str(sorted_data)

    # Generate hash
    return hashlib.sha256(data_str.encode()).hexdigest()


def round_to_tick_size(price: Decimal, tick_size: Decimal) -> Decimal:
    """
    Round price to valid tick size.

    Args:
        price: Price to round
        tick_size: Minimum tick size

    Returns:
        Rounded price
    """
    if tick_size <= 0:
        return price

    return (price / tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick_size


def calculate_notional_value(quantity: Decimal, price: Decimal) -> Decimal:
    """
    Calculate notional value of a position.

    Args:
        quantity: Position quantity
        price: Price per unit

    Returns:
        Notional value
    """
    return abs(quantity) * price


def time_to_market_close(market_timezone: str = "US/Eastern") -> Optional[int]:
    """
    Calculate seconds until market close.

    Args:
        market_timezone: Market timezone

    Returns:
        Seconds until close or None if market closed
    """
    try:
        from datetime import time

        import pytz

        tz = pytz.timezone(market_timezone)
        now = datetime.now(tz)

        # US market hours: 9:30 AM - 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)

        current_time = now.time()

        if market_open <= current_time <= market_close:
            close_datetime = now.replace(
                hour=market_close.hour, minute=market_close.minute, second=0, microsecond=0
            )
            return int((close_datetime - now).total_seconds())

        return None
    except:
        return None


def is_market_open(market_timezone: str = "US/Eastern") -> bool:
    """
    Check if market is currently open.

    Args:
        market_timezone: Market timezone

    Returns:
        True if market is open
    """
    return time_to_market_close(market_timezone) is not None


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def sanitize_string(text: str, max_length: int = 100) -> str:
    """
    Sanitize string for safe storage/display.

    Args:
        text: Text to sanitize
        max_length: Maximum length

    Returns:
        Sanitized string
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove control characters
    sanitized = "".join(char for char in text if ord(char) >= 32)

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[: max_length - 3] + "..."

    return sanitized
