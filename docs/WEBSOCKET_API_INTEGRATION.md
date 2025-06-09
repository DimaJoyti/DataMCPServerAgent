# WebSocket & API Integration Guide

## 🚀 Real-Time Trading System Integration

This document describes the complete WebSocket and API integration for the institutional trading system, providing real-time market data feeds and trading operations.

## 📋 Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React UI      │    │   FastAPI Server │    │  Trading System │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Market Data │◄┼────┼►│ WebSocket    │◄┼────┼►│ Market Data │ │
│ │ Context     │ │    │ │ Endpoints    │ │    │ │ Handler     │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Trading     │◄┼────┼►│ REST API     │◄┼────┼►│ Order       │ │
│ │ Context     │ │    │ │ Endpoints    │ │    │ │ Management  │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Setup Instructions

### 1. Backend Setup

#### Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements-api.txt

# Or using conda
conda install --file requirements-api.txt
```

#### Start the Trading Server
```bash
# Development mode
python start_trading_server.py

# Or directly with uvicorn
uvicorn src.web_interface.trading_api_server:app --reload --host 0.0.0.0 --port 8000
```

#### Server Endpoints
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health
- **Trading WebSocket**: ws://localhost:8000/ws/trading
- **Market Data WebSocket**: ws://localhost:8000/ws/market-data

### 2. Frontend Setup

The React frontend is already configured with the context providers and hooks.

#### Start the Frontend
```bash
cd agent-ui
npm install
npm run dev
```

The UI will be available at http://localhost:3002

## 📡 WebSocket Integration

### Market Data WebSocket

#### Connection
```typescript
import { MarketDataWebSocket } from '@/services/websocket/MarketDataWebSocket'

const marketDataWS = new MarketDataWebSocket({
  url: 'ws://localhost:8000/ws/market-data'
})

await marketDataWS.connect()
```

#### Subscriptions
```typescript
// Subscribe to quotes
marketDataWS.subscribeToQuotes('BTC/USD', (quote) => {
  console.log('Quote update:', quote)
})

// Subscribe to trades
marketDataWS.subscribeToTrades('BTC/USD', (trade) => {
  console.log('Trade update:', trade)
})

// Subscribe to order book
marketDataWS.subscribeToOrderBook('BTC/USD', (orderBook) => {
  console.log('Order book update:', orderBook)
})
```

### Trading WebSocket

#### Authentication
```typescript
import { TradingWebSocket } from '@/services/websocket/TradingWebSocket'

const tradingWS = new TradingWebSocket({
  url: 'ws://localhost:8000/ws/trading'
})

await tradingWS.connect()
await tradingWS.authenticate('demo_api_key', 'signature', Date.now())
```

#### Order Management
```typescript
// Send order
tradingWS.sendOrder({
  symbol: 'BTC/USD',
  side: 'buy',
  type: 'limit',
  quantity: 1.0,
  price: 43000.00
})

// Cancel order
tradingWS.cancelOrder('order-id-123')

// Subscribe to updates
tradingWS.setCallbacks({
  onOrderUpdate: (order) => console.log('Order update:', order),
  onPositionUpdate: (position) => console.log('Position update:', position),
  onTradeExecution: (trade) => console.log('Trade execution:', trade)
})
```

## 🔌 REST API Integration

### Authentication

```typescript
import { tradingAPI } from '@/services/api/TradingAPI'

// Set credentials
tradingAPI.setCredentials('demo_api_key', 'demo_secret')

// Authenticate
const authResponse = await tradingAPI.authenticate()
```

### Order Management

```typescript
// Create order
const order = await tradingAPI.createOrder({
  symbol: 'BTC/USD',
  side: 'buy',
  type: 'limit',
  quantity: 1.0,
  price: 43000.00
})

// Get orders
const orders = await tradingAPI.getOrders()

// Cancel order
await tradingAPI.cancelOrder('order-id-123')
```

### Portfolio Management

```typescript
// Get portfolio
const portfolio = await tradingAPI.getPortfolio()

// Get positions
const positions = await tradingAPI.getPositions()

// Get balances
const balances = await tradingAPI.getBalances()
```

## 🎣 React Hooks Usage

### Market Data Hooks

```typescript
import { useSymbol, useOrderBook, useMarketDataConnection } from '@/services/hooks/useMarketData'

function MarketDataComponent() {
  // Get real-time symbol data
  const { price, change, changePercent, volume } = useSymbol('BTC/USD')
  
  // Get order book
  const { orderBook, loading, error } = useOrderBook('BTC/USD')
  
  // Check connection status
  const { isConnected, connectionState } = useMarketDataConnection()
  
  return (
    <div>
      <p>Price: ${price}</p>
      <p>Change: {change} ({changePercent}%)</p>
      <p>Status: {isConnected ? 'Connected' : 'Disconnected'}</p>
    </div>
  )
}
```

### Trading Hooks

```typescript
import { useCreateOrder, useOrders, usePositions } from '@/services/hooks/useTrading'

function TradingComponent() {
  // Order creation
  const { submitOrder, loading, error } = useCreateOrder()
  
  // Get orders
  const { orders, openOrders, filledOrders } = useOrders()
  
  // Get positions
  const { positions, totalUnrealizedPnL } = usePositions()
  
  const handleCreateOrder = async () => {
    try {
      await submitOrder({
        symbol: 'BTC/USD',
        side: 'buy',
        type: 'market',
        quantity: 1.0
      })
    } catch (err) {
      console.error('Order failed:', err)
    }
  }
  
  return (
    <div>
      <button onClick={handleCreateOrder} disabled={loading}>
        {loading ? 'Submitting...' : 'Buy BTC'}
      </button>
      <p>Open Orders: {openOrders.length}</p>
      <p>Total P&L: ${totalUnrealizedPnL}</p>
    </div>
  )
}
```

## 🔄 Context Providers

### Market Data Provider

```typescript
import { MarketDataProvider } from '@/services/contexts/MarketDataContext'

function App() {
  return (
    <MarketDataProvider 
      wsUrl="ws://localhost:8000/ws/market-data"
      apiUrl="http://localhost:8000/api/v1"
      autoConnect={true}
    >
      <YourTradingComponents />
    </MarketDataProvider>
  )
}
```

### Trading Provider

```typescript
import { TradingProvider } from '@/services/contexts/TradingContext'

function App() {
  return (
    <TradingProvider 
      wsUrl="ws://localhost:8000/ws/trading"
      apiUrl="http://localhost:8000/api/v1"
    >
      <YourTradingComponents />
    </TradingProvider>
  )
}
```

## 🔐 Authentication

### Demo Credentials
For development and testing:
- **API Key**: `demo_api_key`
- **API Secret**: `demo_secret`

### Production Setup
1. Implement proper JWT token generation
2. Add HMAC-SHA256 signature verification
3. Set up user management system
4. Configure rate limiting

## 📊 Data Flow

### Market Data Flow
1. **External Data Sources** → Market Data Handler
2. **Market Data Handler** → WebSocket Server
3. **WebSocket Server** → React Context
4. **React Context** → UI Components

### Trading Flow
1. **UI Components** → Trading Hooks
2. **Trading Hooks** → REST API / WebSocket
3. **API Server** → Trading System
4. **Trading System** → Order Execution
5. **Order Updates** → WebSocket → UI

## 🚨 Error Handling

### WebSocket Reconnection
- Automatic reconnection with exponential backoff
- Subscription restoration on reconnect
- Connection state monitoring

### API Error Handling
- Retry logic for failed requests
- Proper error messages and logging
- Graceful degradation

## 🔧 Configuration

### Environment Variables
```bash
# Server configuration
ENVIRONMENT=development
API_HOST=0.0.0.0
API_PORT=8000

# Database (if using)
DATABASE_URL=postgresql://user:pass@localhost/trading

# External APIs
MARKET_DATA_API_KEY=your_key_here
EXCHANGE_API_KEY=your_key_here
```

## 📈 Performance Optimization

### WebSocket Optimization
- Message batching for high-frequency updates
- Selective subscriptions to reduce bandwidth
- Connection pooling for multiple symbols

### API Optimization
- Response caching for static data
- Pagination for large datasets
- Compression for large responses

## 🧪 Testing

### WebSocket Testing
```bash
# Test WebSocket connection
wscat -c ws://localhost:8000/ws/market-data

# Send test message
{"type": "subscribe", "channel": "quotes", "symbol": "BTC/USD"}
```

### API Testing
```bash
# Test health endpoint
curl http://localhost:8000/api/v1/health

# Test authentication
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"api_key": "demo_api_key", "signature": "test", "timestamp": 1234567890}'
```

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY . .
EXPOSE 8000

CMD ["python", "start_trading_server.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trading-api
  template:
    metadata:
      labels:
        app: trading-api
    spec:
      containers:
      - name: trading-api
        image: trading-api:latest
        ports:
        - containerPort: 8000
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [WebSocket API Reference](https://websockets.readthedocs.io/)
- [React Context Documentation](https://react.dev/reference/react/useContext)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

---

🎉 **Your real-time trading system is now fully integrated with WebSocket and API support!**
