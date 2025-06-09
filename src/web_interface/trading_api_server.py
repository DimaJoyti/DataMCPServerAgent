"""
FastAPI Trading Server
Provides REST API and WebSocket endpoints for the trading system
"""

import asyncio
import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Import trading system components
from ..trading.oms.order_management_system import OrderManagementSystem
from ..trading.core.base_models import BaseOrder, BasePosition, BaseTrade
from ..trading.core.enums import OrderSide, OrderType, OrderStatus, Exchange, Currency
from ..trading.market_data.feed_handler import MockFeedHandler
from ..trading.risk.risk_manager import RiskManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global trading system instance
trading_system: Optional[OrderManagementSystem] = None
risk_manager: Optional[RiskManager] = None
market_data_handler: Optional[MockFeedHandler] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.authenticated_connections: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.authenticated_connections:
            del self.authenticated_connections[websocket]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

    async def broadcast_to_authenticated(self, message: str):
        for connection in self.authenticated_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to authenticated: {e}")

manager = ConnectionManager()

# Pydantic models for API
class OrderRequest(BaseModel):
    symbol: str
    side: str  # 'buy' or 'sell'
    type: str  # 'market', 'limit', 'stop', 'stop-limit'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'day'
    client_order_id: Optional[str] = None

class OrderResponse(BaseModel):
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: str
    type: str
    status: str
    quantity: float
    price: Optional[float]
    filled_quantity: float
    average_fill_price: Optional[float]
    commission: float
    created_at: str
    updated_at: str

class PositionResponse(BaseModel):
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    total_value: float
    open_date: str

class PortfolioResponse(BaseModel):
    total_value: float
    total_pnl: float
    total_pnl_percent: float
    day_pnl: float
    day_pnl_percent: float
    margin_used: float
    margin_available: float
    buying_power: float

class BalanceResponse(BaseModel):
    currency: str
    available: float
    locked: float
    total: float

class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: float
    high_24h: float
    low_24h: float
    bid: float
    ask: float
    timestamp: int

class AuthRequest(BaseModel):
    api_key: str
    signature: str
    timestamp: int

class AuthResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    expires_at: Optional[int] = None
    error: Optional[str] = None

# Authentication
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Simple token verification - in production, use proper JWT validation
    if credentials.credentials != "valid_token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global trading_system, risk_manager, market_data_handler
    
    logger.info("Starting trading system...")
    
    # Initialize trading system components
    trading_system = OrderManagementSystem(
        name="WebTradingSystem",
        enable_smart_routing=True,
        enable_algorithms=True,
        max_orders_per_second=1000
    )
    
    risk_manager = RiskManager()
    market_data_handler = MockFeedHandler()
    
    # Start the trading system
    await trading_system.start()
    await market_data_handler.start()
    
    logger.info("Trading system started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down trading system...")
    if trading_system:
        await trading_system.stop()
    if market_data_handler:
        await market_data_handler.stop()
    logger.info("Trading system shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Institutional Trading System API",
    description="High-performance trading system API with WebSocket support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication endpoints
@app.post("/api/v1/auth/login", response_model=AuthResponse)
async def login(auth_request: AuthRequest):
    """Authenticate user and return access token"""
    try:
        # Simple authentication - in production, verify signature properly
        if auth_request.api_key == "demo_api_key":
            return AuthResponse(
                success=True,
                token="valid_token",
                expires_at=int(datetime.now().timestamp()) + 3600
            )
        else:
            return AuthResponse(
                success=False,
                error="Invalid API key"
            )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return AuthResponse(
            success=False,
            error="Authentication failed"
        )

# Order management endpoints
@app.post("/api/v1/orders", response_model=OrderResponse)
async def create_order(order_request: OrderRequest, token: str = Depends(verify_token)):
    """Create a new order"""
    try:
        if not trading_system:
            raise HTTPException(status_code=503, detail="Trading system not available")
        
        # Convert request to BaseOrder
        order = BaseOrder(
            symbol=order_request.symbol,
            side=OrderSide.BUY if order_request.side.lower() == 'buy' else OrderSide.SELL,
            order_type=OrderType[order_request.type.upper()],
            quantity=Decimal(str(order_request.quantity)),
            price=Decimal(str(order_request.price)) if order_request.price else None,
            stop_price=Decimal(str(order_request.stop_price)) if order_request.stop_price else None,
            client_order_id=order_request.client_order_id
        )
        
        # Submit order
        order_id = await trading_system.submit_order(order)
        
        # Get the created order
        created_order = trading_system.get_order(order_id)
        
        # Broadcast order update via WebSocket
        await manager.broadcast_to_authenticated(json.dumps({
            "type": "order_update",
            "data": {
                "orderId": order_id,
                "symbol": order_request.symbol,
                "side": order_request.side,
                "type": order_request.type,
                "status": "pending",
                "quantity": order_request.quantity,
                "price": order_request.price,
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
        }))
        
        return OrderResponse(
            order_id=order_id,
            client_order_id=order_request.client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            type=order_request.type,
            status="pending",
            quantity=order_request.quantity,
            price=order_request.price,
            filled_quantity=0.0,
            average_fill_price=None,
            commission=0.0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/orders", response_model=List[OrderResponse])
async def get_orders(token: str = Depends(verify_token)):
    """Get all orders"""
    try:
        if not trading_system:
            raise HTTPException(status_code=503, detail="Trading system not available")
        
        orders = []
        for order_id, order in trading_system.orders.items():
            orders.append(OrderResponse(
                order_id=order_id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side.value,
                type=order.order_type.value,
                status=order.status.value,
                quantity=float(order.quantity),
                price=float(order.price) if order.price else None,
                filled_quantity=float(order.filled_quantity),
                average_fill_price=float(order.average_fill_price) if order.average_fill_price else None,
                commission=float(order.commission),
                created_at=order.created_at.isoformat(),
                updated_at=order.updated_at.isoformat()
            ))
        
        return orders
        
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/orders/{order_id}")
async def cancel_order(order_id: str, token: str = Depends(verify_token)):
    """Cancel an order"""
    try:
        if not trading_system:
            raise HTTPException(status_code=503, detail="Trading system not available")
        
        success = await trading_system.cancel_order(order_id)
        
        if success:
            # Broadcast order cancellation via WebSocket
            await manager.broadcast_to_authenticated(json.dumps({
                "type": "order_update",
                "data": {
                    "orderId": order_id,
                    "status": "cancelled",
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }
            }))
            
            return {"success": True, "message": "Order cancelled successfully"}
        else:
            raise HTTPException(status_code=404, detail="Order not found")
            
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Position endpoints
@app.get("/api/v1/positions", response_model=List[PositionResponse])
async def get_positions(token: str = Depends(verify_token)):
    """Get all positions"""
    try:
        # Mock positions data - integrate with actual position manager
        positions = [
            PositionResponse(
                symbol="BTC/USD",
                side="long",
                quantity=2.5,
                entry_price=42800.00,
                current_price=43250.75,
                unrealized_pnl=1126.88,
                realized_pnl=0.0,
                total_value=108126.88,
                open_date=datetime.now().isoformat()
            )
        ]
        
        return positions
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Portfolio endpoint
@app.get("/api/v1/portfolio", response_model=PortfolioResponse)
async def get_portfolio(token: str = Depends(verify_token)):
    """Get portfolio summary"""
    try:
        # Mock portfolio data - integrate with actual portfolio manager
        return PortfolioResponse(
            total_value=223954.38,
            total_pnl=149.38,
            total_pnl_percent=0.067,
            day_pnl=1247.25,
            day_pnl_percent=0.56,
            margin_used=45000.00,
            margin_available=155000.00,
            buying_power=400000.00
        )
        
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Market data endpoints
@app.get("/api/v1/market-data/{symbol}", response_model=MarketDataResponse)
async def get_market_data(symbol: str):
    """Get market data for a symbol"""
    try:
        # Mock market data - integrate with actual market data feed
        return MarketDataResponse(
            symbol=symbol,
            price=43250.75,
            change=1250.25,
            change_percent=2.98,
            volume=125000000,
            high_24h=44100.00,
            low_24h=41800.50,
            bid=43248.50,
            ask=43252.25,
            timestamp=int(datetime.now().timestamp() * 1000)
        )
        
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoints
@app.websocket("/ws/trading")
async def trading_websocket(websocket: WebSocket):
    """WebSocket endpoint for trading updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "auth":
                # Handle authentication
                api_key = message.get("apiKey")
                if api_key == "demo_api_key":
                    manager.authenticated_connections[websocket] = api_key
                    await manager.send_personal_message(json.dumps({
                        "type": "auth_response",
                        "success": True
                    }), websocket)
                else:
                    await manager.send_personal_message(json.dumps({
                        "type": "auth_response",
                        "success": False,
                        "error": "Invalid API key"
                    }), websocket)
            
            elif message.get("type") == "ping":
                # Handle heartbeat
                await manager.send_personal_message(json.dumps({
                    "type": "pong",
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/market-data")
async def market_data_websocket(websocket: WebSocket):
    """WebSocket endpoint for market data updates"""
    await manager.connect(websocket)
    try:
        # Send periodic market data updates
        while True:
            # Mock market data update
            market_update = {
                "type": "ticker",
                "symbol": "BTC/USD",
                "price": 43250.75 + (asyncio.get_event_loop().time() % 100 - 50),
                "change": 1250.25,
                "changePercent": 2.98,
                "volume": 125000000,
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
            
            await manager.send_personal_message(json.dumps(market_update), websocket)
            await asyncio.sleep(1)  # Send updates every second
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Health check endpoint
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "trading_system": trading_system is not None,
        "active_connections": len(manager.active_connections)
    }

if __name__ == "__main__":
    uvicorn.run(
        "trading_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
