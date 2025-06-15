"""
Core enums for the institutional trading system.
"""

from enum import Enum
from typing import Dict


class OrderType(Enum):
    """Order types supported by the trading system."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"
    IMPLEMENTATION_SHORTFALL = "IMPLEMENTATION_SHORTFALL"
    ARRIVAL_PRICE = "ARRIVAL_PRICE"
    PERCENT_OF_VOLUME = "PERCENT_OF_VOLUME"


class OrderSide(Enum):
    """Order side (buy/sell)."""

    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class OrderStatus(Enum):
    """Order status lifecycle."""

    PENDING = "PENDING"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    SUSPENDED = "SUSPENDED"


class TimeInForce(Enum):
    """Time in force options."""

    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date
    ATO = "ATO"  # At The Open
    ATC = "ATC"  # At The Close


class AssetClass(Enum):
    """Asset classes supported by the system."""

    EQUITY = "EQUITY"
    FIXED_INCOME = "FIXED_INCOME"
    COMMODITY = "COMMODITY"
    CURRENCY = "CURRENCY"
    CRYPTOCURRENCY = "CRYPTOCURRENCY"
    DERIVATIVE = "DERIVATIVE"
    ALTERNATIVE = "ALTERNATIVE"


class Exchange(Enum):
    """Supported exchanges."""

    # Equity Exchanges
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    LSE = "LSE"
    TSE = "TSE"
    HKEX = "HKEX"

    # Crypto Exchanges
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"
    KRAKEN = "KRAKEN"
    BITFINEX = "BITFINEX"
    HUOBI = "HUOBI"

    # FX Exchanges
    EBS = "EBS"
    REUTERS = "REUTERS"
    CURRENEX = "CURRENEX"

    # Futures Exchanges
    CME = "CME"
    ICE = "ICE"
    EUREX = "EUREX"


class Currency(Enum):
    """Supported currencies."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    CNY = "CNY"
    HKD = "HKD"
    SGD = "SGD"

    # Cryptocurrencies
    BTC = "BTC"
    ETH = "ETH"
    USDT = "USDT"
    USDC = "USDC"


class RiskLevel(Enum):
    """Risk levels for positions and strategies."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class StrategyType(Enum):
    """Trading strategy types."""

    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    ARBITRAGE = "ARBITRAGE"
    MARKET_MAKING = "MARKET_MAKING"
    STATISTICAL_ARBITRAGE = "STATISTICAL_ARBITRAGE"
    PAIRS_TRADING = "PAIRS_TRADING"
    TREND_FOLLOWING = "TREND_FOLLOWING"
    VOLATILITY = "VOLATILITY"
    ALPHA_CAPTURE = "ALPHA_CAPTURE"


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""

    TWAP = "TWAP"  # Time Weighted Average Price
    VWAP = "VWAP"  # Volume Weighted Average Price
    IMPLEMENTATION_SHORTFALL = "IMPLEMENTATION_SHORTFALL"
    ARRIVAL_PRICE = "ARRIVAL_PRICE"
    PERCENT_OF_VOLUME = "PERCENT_OF_VOLUME"
    ICEBERG = "ICEBERG"
    SNIPER = "SNIPER"
    GUERRILLA = "GUERRILLA"


class MarketDataType(Enum):
    """Market data types."""

    TICK = "TICK"
    QUOTE = "QUOTE"
    TRADE = "TRADE"
    DEPTH = "DEPTH"
    OHLCV = "OHLCV"
    NEWS = "NEWS"
    FUNDAMENTAL = "FUNDAMENTAL"


class SystemStatus(Enum):
    """System status levels."""

    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    MAINTENANCE = "MAINTENANCE"


# Exchange-specific configurations
EXCHANGE_CONFIGS: Dict[Exchange, Dict] = {
    Exchange.BINANCE: {
        "asset_classes": [AssetClass.CRYPTOCURRENCY],
        "supported_order_types": [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LIMIT],
        "min_order_size": 0.001,
        "max_order_size": 1000000,
        "tick_size": 0.01,
        "trading_hours": "24/7",
    },
    Exchange.NYSE: {
        "asset_classes": [AssetClass.EQUITY],
        "supported_order_types": [
            OrderType.MARKET,
            OrderType.LIMIT,
            OrderType.STOP,
            OrderType.STOP_LIMIT,
        ],
        "min_order_size": 1,
        "max_order_size": 1000000,
        "tick_size": 0.01,
        "trading_hours": "09:30-16:00 EST",
    },
    Exchange.CME: {
        "asset_classes": [AssetClass.DERIVATIVE, AssetClass.COMMODITY],
        "supported_order_types": [
            OrderType.MARKET,
            OrderType.LIMIT,
            OrderType.STOP,
            OrderType.ICEBERG,
        ],
        "min_order_size": 1,
        "max_order_size": 10000,
        "tick_size": 0.25,
        "trading_hours": "17:00-16:00 CT",
    },
}

# Asset class configurations
ASSET_CLASS_CONFIGS: Dict[AssetClass, Dict] = {
    AssetClass.EQUITY: {
        "settlement_period": "T+2",
        "margin_requirement": 0.25,
        "short_selling_allowed": True,
        "fractional_shares": True,
    },
    AssetClass.CRYPTOCURRENCY: {
        "settlement_period": "T+0",
        "margin_requirement": 0.5,
        "short_selling_allowed": True,
        "fractional_shares": True,
    },
    AssetClass.FIXED_INCOME: {
        "settlement_period": "T+1",
        "margin_requirement": 0.1,
        "short_selling_allowed": False,
        "fractional_shares": False,
    },
}
