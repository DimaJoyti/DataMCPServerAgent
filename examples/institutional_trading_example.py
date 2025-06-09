#!/usr/bin/env python3
"""
Institutional Trading System Example

Demonstrates the enhanced Order Management System (OMS) with:
- High-frequency trading capabilities
- Smart order routing
- Execution algorithms (TWAP, VWAP, Implementation Shortfall)
- Real-time risk management
- Multi-strategy execution
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading.core.enums import OrderSide, OrderType, Exchange, Currency
from trading.oms.order_management_system import OrderManagementSystem
from trading.oms.order_types import (
    create_twap_order, create_vwap_order, create_iceberg_order,
    TWAPOrder, VWAPOrder, IcebergOrder
)
from trading.core.base_models import BaseOrder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("InstitutionalTradingExample")


async def demo_basic_order_management():
    """Demonstrate basic order management functionality."""
    print("\n" + "="*60)
    print("🏦 INSTITUTIONAL TRADING SYSTEM DEMO")
    print("="*60)
    
    # Initialize OMS
    oms = OrderManagementSystem(
        name="HedgeFundOMS",
        enable_smart_routing=True,
        enable_algorithms=True,
        max_orders_per_second=10000,
        latency_threshold_ms=1.0
    )
    
    await oms.start()
    
    print("\n📊 Order Management System Status:")
    print(f"   ✅ OMS Name: {oms.name}")
    print(f"   ✅ Smart Routing: {'Enabled' if oms.enable_smart_routing else 'Disabled'}")
    print(f"   ✅ Algorithms: {'Enabled' if oms.enable_algorithms else 'Disabled'}")
    print(f"   ✅ Max Orders/sec: {oms.max_orders_per_second:,}")
    print(f"   ✅ Latency Threshold: {oms.latency_threshold_ms}ms")
    
    # Create sample orders
    orders = []
    
    # 1. Simple market order
    market_order = BaseOrder(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('1000'),
        exchange=Exchange.NASDAQ,
        currency=Currency.USD,
        strategy_id="MOMENTUM_001",
        portfolio_id="TECH_PORTFOLIO"
    )
    orders.append(("Market Order", market_order))
    
    # 2. Limit order
    limit_order = BaseOrder(
        symbol="MSFT",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal('500'),
        price=Decimal('350.00'),
        exchange=Exchange.NASDAQ,
        currency=Currency.USD,
        strategy_id="MEAN_REVERSION_002",
        portfolio_id="TECH_PORTFOLIO"
    )
    orders.append(("Limit Order", limit_order))
    
    # 3. Stop-loss order
    stop_order = BaseOrder(
        symbol="GOOGL",
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=Decimal('200'),
        stop_price=Decimal('2800.00'),
        exchange=Exchange.NASDAQ,
        currency=Currency.USD,
        strategy_id="RISK_MANAGEMENT_003",
        portfolio_id="TECH_PORTFOLIO"
    )
    orders.append(("Stop Order", stop_order))
    
    # Submit orders
    print("\n📤 Submitting Orders:")
    submitted_orders = []
    
    for order_name, order in orders:
        try:
            order_id = await oms.submit_order(order)
            submitted_orders.append(order_id)
            print(f"   ✅ {order_name}: {order_id} ({order.symbol})")
        except Exception as e:
            print(f"   ❌ {order_name}: Failed - {str(e)}")
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check order status
    print("\n📋 Order Status:")
    for order_id in submitted_orders:
        order = oms.get_order(order_id)
        if order:
            print(f"   📊 {order_id}: {order.status.value} - {order.symbol} {order.quantity}")
    
    # Get performance metrics
    metrics = await oms.get_performance_metrics()
    print("\n📈 OMS Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   📊 {key}: {value:.2f}")
        else:
            print(f"   📊 {key}: {value}")
    
    await oms.stop()
    return oms


async def demo_algorithmic_orders():
    """Demonstrate algorithmic order execution."""
    print("\n" + "="*60)
    print("🤖 ALGORITHMIC EXECUTION DEMO")
    print("="*60)
    
    # Initialize OMS
    oms = OrderManagementSystem(
        name="AlgoTradingOMS",
        enable_algorithms=True
    )
    
    await oms.start()
    
    # 1. TWAP Order
    print("\n⏰ Creating TWAP Order:")
    twap_order = create_twap_order(
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=Decimal('10000'),
        duration_hours=2.0,
        slice_interval_minutes=10
    )
    
    print(f"   📊 Symbol: {twap_order.symbol}")
    print(f"   📊 Quantity: {twap_order.quantity:,}")
    print(f"   📊 Duration: {(twap_order.end_time - twap_order.start_time).total_seconds() / 3600:.1f} hours")
    print(f"   📊 Slices: {twap_order.total_slices}")
    print(f"   📊 Slice Interval: {twap_order.slice_interval}")
    
    try:
        twap_id = await oms.submit_order(twap_order)
        print(f"   ✅ TWAP Order Submitted: {twap_id}")
    except Exception as e:
        print(f"   ❌ TWAP Order Failed: {str(e)}")
    
    # 2. VWAP Order
    print("\n📊 Creating VWAP Order:")
    vwap_order = create_vwap_order(
        symbol="QQQ",
        side=OrderSide.SELL,
        quantity=Decimal('5000'),
        duration_hours=1.5,
        max_participation=0.15
    )
    
    print(f"   📊 Symbol: {vwap_order.symbol}")
    print(f"   📊 Quantity: {vwap_order.quantity:,}")
    print(f"   📊 Max Participation: {vwap_order.max_participation_rate:.1%}")
    
    try:
        vwap_id = await oms.submit_order(vwap_order)
        print(f"   ✅ VWAP Order Submitted: {vwap_id}")
    except Exception as e:
        print(f"   ❌ VWAP Order Failed: {str(e)}")
    
    # 3. Iceberg Order
    print("\n🧊 Creating Iceberg Order:")
    iceberg_order = create_iceberg_order(
        symbol="IWM",
        side=OrderSide.BUY,
        quantity=Decimal('20000'),
        price=Decimal('220.00'),
        display_percentage=0.05
    )
    
    print(f"   📊 Symbol: {iceberg_order.symbol}")
    print(f"   📊 Total Quantity: {iceberg_order.quantity:,}")
    print(f"   📊 Display Quantity: {iceberg_order.display_quantity:,}")
    print(f"   📊 Hidden Quantity: {iceberg_order.hidden_quantity:,}")
    print(f"   📊 Total Slices: {iceberg_order.total_slices}")
    
    try:
        iceberg_id = await oms.submit_order(iceberg_order)
        print(f"   ✅ Iceberg Order Submitted: {iceberg_id}")
    except Exception as e:
        print(f"   ❌ Iceberg Order Failed: {str(e)}")
    
    # Monitor execution for a short time
    print("\n⏳ Monitoring Execution (5 seconds)...")
    await asyncio.sleep(5)
    
    # Check final status
    print("\n📋 Final Order Status:")
    for order_id in [twap_id, vwap_id, iceberg_id]:
        if order_id:
            order = oms.get_order(order_id)
            if order:
                print(f"   📊 {order_id}: {order.status.value} - Progress: {getattr(order, 'execution_progress', 0):.1%}")
    
    await oms.stop()


async def demo_smart_routing():
    """Demonstrate smart order routing."""
    print("\n" + "="*60)
    print("🧠 SMART ORDER ROUTING DEMO")
    print("="*60)
    
    # Initialize OMS with smart routing
    oms = OrderManagementSystem(
        name="SmartRoutingOMS",
        enable_smart_routing=True
    )
    
    await oms.start()
    
    if oms.smart_router:
        # Get venue status
        venue_status = oms.smart_router.get_venue_status()
        print("\n🏦 Available Trading Venues:")
        for exchange, status in venue_status.items():
            print(f"   📊 {exchange.value}:")
            print(f"      ⚡ Latency: {status['latency_ms']}ms")
            print(f"      🔒 Reliability: {status['reliability']:.1%}")
            print(f"      💰 Fee Rate: {status['fee_rate']:.2%}")
            print(f"      📡 Status: {status['status']}")
    
    # Create orders for routing
    routing_orders = [
        BaseOrder(
            symbol="TSLA",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('1000'),
            strategy_id="MOMENTUM_ROUTING"
        ),
        BaseOrder(
            symbol="NVDA",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal('500'),
            price=Decimal('800.00'),
            strategy_id="ARBITRAGE_ROUTING"
        ),
        BaseOrder(
            symbol="AMD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('2000'),
            strategy_id="LARGE_ORDER_ROUTING"
        )
    ]
    
    print("\n📤 Submitting Orders for Smart Routing:")
    for i, order in enumerate(routing_orders, 1):
        try:
            order_id = await oms.submit_order(order)
            print(f"   ✅ Order {i}: {order_id} ({order.symbol}) - Routed via Smart Router")
        except Exception as e:
            print(f"   ❌ Order {i}: Failed - {str(e)}")
    
    # Get routing statistics
    if oms.smart_router:
        routing_stats = oms.smart_router.get_routing_statistics()
        print("\n📊 Smart Routing Statistics:")
        for key, value in routing_stats.items():
            if isinstance(value, float):
                print(f"   📊 {key}: {value:.2f}")
            else:
                print(f"   📊 {key}: {value}")
    
    await oms.stop()


async def demo_risk_management():
    """Demonstrate risk management features."""
    print("\n" + "="*60)
    print("⚠️ RISK MANAGEMENT DEMO")
    print("="*60)
    
    # Initialize OMS
    oms = OrderManagementSystem(
        name="RiskManagedOMS",
        max_orders_per_second=100,  # Lower limit for demo
        latency_threshold_ms=0.5
    )
    
    # Set position limits
    oms.position_limits["AAPL"] = Decimal('5000')
    oms.position_limits["MSFT"] = Decimal('3000')
    
    await oms.start()
    
    print("\n🛡️ Risk Limits Configuration:")
    print(f"   📊 Daily Order Limit: {oms.daily_order_limit:,}")
    print(f"   📊 Daily Notional Limit: ${oms.daily_notional_limit:,}")
    print(f"   📊 Max Orders/Second: {oms.max_orders_per_second}")
    print(f"   📊 Latency Threshold: {oms.latency_threshold_ms}ms")
    print(f"   📊 Position Limits:")
    for symbol, limit in oms.position_limits.items():
        print(f"      📊 {symbol}: {limit:,} shares")
    
    # Test risk limits
    print("\n🧪 Testing Risk Limits:")
    
    # 1. Normal order (should pass)
    normal_order = BaseOrder(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('100'),
        price=Decimal('150.00')
    )
    
    try:
        order_id = await oms.submit_order(normal_order)
        print(f"   ✅ Normal Order: {order_id} - Passed risk checks")
    except Exception as e:
        print(f"   ❌ Normal Order: Failed - {str(e)}")
    
    # 2. Large order (might trigger warnings)
    large_order = BaseOrder(
        symbol="MSFT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('10000'),
        price=Decimal('350.00')
    )
    
    try:
        order_id = await oms.submit_order(large_order)
        print(f"   ⚠️ Large Order: {order_id} - Passed with warnings")
    except Exception as e:
        print(f"   ❌ Large Order: Failed - {str(e)}")
    
    await oms.stop()


async def demo_performance_monitoring():
    """Demonstrate performance monitoring."""
    print("\n" + "="*60)
    print("📈 PERFORMANCE MONITORING DEMO")
    print("="*60)
    
    # Initialize high-performance OMS
    oms = OrderManagementSystem(
        name="HighPerfOMS",
        max_orders_per_second=50000,
        latency_threshold_ms=0.1
    )
    
    await oms.start()
    
    # Submit multiple orders rapidly
    print("\n🚀 High-Frequency Order Submission Test:")
    start_time = datetime.utcnow()
    
    order_count = 100
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    for i in range(order_count):
        symbol = symbols[i % len(symbols)]
        order = BaseOrder(
            symbol=symbol,
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal('100'),
            strategy_id=f"HFT_STRATEGY_{i % 10}"
        )
        
        try:
            await oms.submit_order(order)
        except Exception as e:
            print(f"   ❌ Order {i}: Failed - {str(e)}")
    
    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()
    
    print(f"   📊 Orders Submitted: {order_count}")
    print(f"   📊 Duration: {duration:.3f} seconds")
    print(f"   📊 Orders/Second: {order_count / duration:.0f}")
    
    # Get final performance metrics
    metrics = await oms.get_performance_metrics()
    print("\n📊 Final Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   📊 {key}: {value:.2f}")
        else:
            print(f"   📊 {key}: {value:,}")
    
    await oms.stop()


async def main():
    """Run all trading system demos."""
    try:
        print("🏦 INSTITUTIONAL TRADING SYSTEM DEMONSTRATION")
        print("=" * 80)
        print("This demo showcases enterprise-grade trading capabilities:")
        print("• High-frequency order management")
        print("• Smart order routing")
        print("• Execution algorithms (TWAP, VWAP, Iceberg)")
        print("• Real-time risk management")
        print("• Performance monitoring")
        print("=" * 80)
        
        # Run demos
        await demo_basic_order_management()
        await demo_algorithmic_orders()
        await demo_smart_routing()
        await demo_risk_management()
        await demo_performance_monitoring()
        
        print("\n" + "="*60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n🚀 Ready for institutional trading operations!")
        print("💡 Next steps:")
        print("   1. Integrate with real exchange APIs")
        print("   2. Connect to market data feeds")
        print("   3. Implement custom strategies")
        print("   4. Set up monitoring and alerting")
        print("   5. Configure risk management rules")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
