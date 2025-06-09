'use client'

/**
 * Trading Context Provider
 * Manages trading state, orders, positions, and WebSocket connections
 */

import React, { createContext, useContext, useReducer, useEffect, useCallback, ReactNode } from 'react'
import { TradingWebSocket, OrderUpdate, PositionUpdate, BalanceUpdate, TradeExecution, RiskAlert } from '../websocket/TradingWebSocket'
import { TradingAPI, tradingAPI } from '../api/TradingAPI'
import { Order, Position, Portfolio, Balance, CreateOrderRequest } from '../api/types'

// State Types
interface TradingState {
  orders: Record<string, Order>
  positions: Record<string, Position>
  balances: Record<string, Balance>
  portfolio: Portfolio | null
  recentTrades: TradeExecution[]
  riskAlerts: RiskAlert[]
  connectionState: 'disconnected' | 'connecting' | 'connected' | 'error'
  authenticated: boolean
  lastUpdate: number
}

// Action Types
type TradingAction =
  | { type: 'SET_CONNECTION_STATE'; payload: TradingState['connectionState'] }
  | { type: 'SET_AUTHENTICATED'; payload: boolean }
  | { type: 'UPDATE_ORDER'; payload: OrderUpdate }
  | { type: 'SET_ORDERS'; payload: Order[] }
  | { type: 'UPDATE_POSITION'; payload: PositionUpdate }
  | { type: 'SET_POSITIONS'; payload: Position[] }
  | { type: 'UPDATE_BALANCE'; payload: BalanceUpdate }
  | { type: 'SET_BALANCES'; payload: Balance[] }
  | { type: 'SET_PORTFOLIO'; payload: Portfolio }
  | { type: 'ADD_TRADE_EXECUTION'; payload: TradeExecution }
  | { type: 'ADD_RISK_ALERT'; payload: RiskAlert }
  | { type: 'ACKNOWLEDGE_RISK_ALERT'; payload: string }
  | { type: 'CLEAR_DATA' }

// Context Type
interface TradingContextType {
  state: TradingState
  // Order Management
  createOrder: (order: CreateOrderRequest) => Promise<Order>
  cancelOrder: (orderId: string) => Promise<void>
  modifyOrder: (orderId: string, updates: any) => Promise<Order>
  refreshOrders: () => Promise<void>
  // Position Management
  getPosition: (symbol: string) => Position | undefined
  closePosition: (symbol: string, quantity?: number) => Promise<void>
  refreshPositions: () => Promise<void>
  // Portfolio Management
  refreshPortfolio: () => Promise<void>
  refreshBalances: () => Promise<void>
  // Risk Management
  acknowledgeRiskAlert: (alertId: string) => void
  getActiveRiskAlerts: () => RiskAlert[]
  // Connection Management
  connect: () => Promise<void>
  disconnect: () => void
  authenticate: (apiKey: string, apiSecret: string) => Promise<void>
  isConnected: () => boolean
  isAuthenticated: () => boolean
}

// Initial State
const initialState: TradingState = {
  orders: {},
  positions: {},
  balances: {},
  portfolio: null,
  recentTrades: [],
  riskAlerts: [],
  connectionState: 'disconnected',
  authenticated: false,
  lastUpdate: 0
}

// Reducer
function tradingReducer(state: TradingState, action: TradingAction): TradingState {
  switch (action.type) {
    case 'SET_CONNECTION_STATE':
      return {
        ...state,
        connectionState: action.payload
      }

    case 'SET_AUTHENTICATED':
      return {
        ...state,
        authenticated: action.payload
      }

    case 'UPDATE_ORDER':
      const orderUpdate = action.payload
      const existingOrder = state.orders[orderUpdate.orderId]
      
      const updatedOrder: Order = {
        orderId: orderUpdate.orderId,
        clientOrderId: orderUpdate.clientOrderId,
        symbol: orderUpdate.symbol,
        side: orderUpdate.side,
        type: orderUpdate.type,
        status: orderUpdate.status,
        quantity: orderUpdate.quantity,
        price: orderUpdate.price,
        filledQuantity: orderUpdate.filledQuantity,
        averageFillPrice: orderUpdate.averageFillPrice,
        commission: 0, // Will be updated from trade executions
        timeInForce: 'day', // Default
        createdAt: existingOrder?.createdAt || new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }

      return {
        ...state,
        orders: {
          ...state.orders,
          [orderUpdate.orderId]: updatedOrder
        },
        lastUpdate: Date.now()
      }

    case 'SET_ORDERS':
      const ordersMap = action.payload.reduce((acc, order) => {
        acc[order.orderId] = order
        return acc
      }, {} as Record<string, Order>)

      return {
        ...state,
        orders: ordersMap,
        lastUpdate: Date.now()
      }

    case 'UPDATE_POSITION':
      const positionUpdate = action.payload
      const updatedPosition: Position = {
        symbol: positionUpdate.symbol,
        side: positionUpdate.side,
        quantity: positionUpdate.quantity,
        entryPrice: positionUpdate.entryPrice,
        currentPrice: positionUpdate.currentPrice,
        unrealizedPnL: positionUpdate.unrealizedPnL,
        realizedPnL: positionUpdate.realizedPnL,
        totalValue: positionUpdate.quantity * positionUpdate.currentPrice,
        riskLevel: 'medium', // Default
        openDate: new Date().toISOString()
      }

      return {
        ...state,
        positions: {
          ...state.positions,
          [positionUpdate.symbol]: updatedPosition
        },
        lastUpdate: Date.now()
      }

    case 'SET_POSITIONS':
      const positionsMap = action.payload.reduce((acc, position) => {
        acc[position.symbol] = position
        return acc
      }, {} as Record<string, Position>)

      return {
        ...state,
        positions: positionsMap,
        lastUpdate: Date.now()
      }

    case 'UPDATE_BALANCE':
      const balanceUpdate = action.payload
      const updatedBalance: Balance = {
        currency: balanceUpdate.currency,
        available: balanceUpdate.available,
        locked: balanceUpdate.locked,
        total: balanceUpdate.total
      }

      return {
        ...state,
        balances: {
          ...state.balances,
          [balanceUpdate.currency]: updatedBalance
        },
        lastUpdate: Date.now()
      }

    case 'SET_BALANCES':
      const balancesMap = action.payload.reduce((acc, balance) => {
        acc[balance.currency] = balance
        return acc
      }, {} as Record<string, Balance>)

      return {
        ...state,
        balances: balancesMap,
        lastUpdate: Date.now()
      }

    case 'SET_PORTFOLIO':
      return {
        ...state,
        portfolio: action.payload,
        lastUpdate: Date.now()
      }

    case 'ADD_TRADE_EXECUTION':
      const newTrades = [action.payload, ...state.recentTrades].slice(0, 100) // Keep last 100 trades
      
      return {
        ...state,
        recentTrades: newTrades,
        lastUpdate: Date.now()
      }

    case 'ADD_RISK_ALERT':
      return {
        ...state,
        riskAlerts: [action.payload, ...state.riskAlerts],
        lastUpdate: Date.now()
      }

    case 'ACKNOWLEDGE_RISK_ALERT':
      return {
        ...state,
        riskAlerts: state.riskAlerts.filter(alert => alert.id !== action.payload),
        lastUpdate: Date.now()
      }

    case 'CLEAR_DATA':
      return {
        ...initialState,
        connectionState: state.connectionState
      }

    default:
      return state
  }
}

// Create Context
const TradingContext = createContext<TradingContextType | undefined>(undefined)

// Provider Props
interface TradingProviderProps {
  children: ReactNode
  wsUrl?: string
  apiUrl?: string
  autoConnect?: boolean
}

// Provider Component
export const TradingProvider: React.FC<TradingProviderProps> = ({
  children,
  wsUrl = 'ws://localhost:8765/ws/trading',
  apiUrl = 'http://localhost:8000/api/v1',
  autoConnect = false
}) => {
  const [state, dispatch] = useReducer(tradingReducer, initialState)
  const [wsClient, setWsClient] = React.useState<TradingWebSocket | null>(null)
  const [apiClient] = React.useState(() => new TradingAPI(apiUrl))

  // Initialize WebSocket connection
  useEffect(() => {
    const ws = new TradingWebSocket({ url: wsUrl })
    setWsClient(ws)

    // Setup event handlers
    ws.addEventListener('stateChange', (state) => {
      dispatch({ type: 'SET_CONNECTION_STATE', payload: state as any })
    })

    ws.addEventListener('error', (error) => {
      console.error('Trading WebSocket error:', error)
      dispatch({ type: 'SET_CONNECTION_STATE', payload: 'error' })
    })

    // Setup trading callbacks
    ws.setCallbacks({
      onOrderUpdate: (order) => {
        dispatch({ type: 'UPDATE_ORDER', payload: order })
      },
      onPositionUpdate: (position) => {
        dispatch({ type: 'UPDATE_POSITION', payload: position })
      },
      onBalanceUpdate: (balance) => {
        dispatch({ type: 'UPDATE_BALANCE', payload: balance })
      },
      onTradeExecution: (trade) => {
        dispatch({ type: 'ADD_TRADE_EXECUTION', payload: trade })
      },
      onRiskAlert: (alert) => {
        dispatch({ type: 'ADD_RISK_ALERT', payload: alert })
      }
    })

    if (autoConnect) {
      ws.connect().catch(console.error)
    }

    return () => {
      ws.disconnect()
    }
  }, [wsUrl, autoConnect])

  // Connection Management
  const connect = useCallback(async () => {
    if (wsClient) {
      await wsClient.connect()
    }
  }, [wsClient])

  const disconnect = useCallback(() => {
    if (wsClient) {
      wsClient.disconnect()
      dispatch({ type: 'SET_AUTHENTICATED', payload: false })
    }
  }, [wsClient])

  const authenticate = useCallback(async (apiKey: string, apiSecret: string) => {
    try {
      // Authenticate with REST API
      apiClient.setCredentials(apiKey, apiSecret)
      await apiClient.authenticate()

      // Authenticate with WebSocket
      if (wsClient) {
        const timestamp = Date.now()
        const signature = btoa(`${timestamp}${apiKey}`) // Mock signature
        await wsClient.authenticate(apiKey, signature, timestamp)
        
        // Subscribe to all trading updates
        wsClient.subscribeToAll()
        
        dispatch({ type: 'SET_AUTHENTICATED', payload: true })
        
        // Load initial data
        await Promise.all([
          refreshOrders(),
          refreshPositions(),
          refreshPortfolio(),
          refreshBalances()
        ])
      }
    } catch (error) {
      console.error('Authentication failed:', error)
      throw error
    }
  }, [wsClient, apiClient])

  // Order Management
  const createOrder = useCallback(async (order: CreateOrderRequest): Promise<Order> => {
    const createdOrder = await apiClient.createOrder(order)
    
    // Also send via WebSocket for real-time updates
    if (wsClient && wsClient.isReady()) {
      wsClient.sendOrder(order)
    }
    
    return createdOrder
  }, [apiClient, wsClient])

  const cancelOrder = useCallback(async (orderId: string): Promise<void> => {
    await apiClient.cancelOrder(orderId)
    
    // Also send via WebSocket
    if (wsClient && wsClient.isReady()) {
      wsClient.cancelOrder(orderId)
    }
  }, [apiClient, wsClient])

  const modifyOrder = useCallback(async (orderId: string, updates: any): Promise<Order> => {
    const modifiedOrder = await apiClient.modifyOrder({ orderId, ...updates })
    
    // Also send via WebSocket
    if (wsClient && wsClient.isReady()) {
      wsClient.modifyOrder(orderId, updates)
    }
    
    return modifiedOrder
  }, [apiClient, wsClient])

  const refreshOrders = useCallback(async () => {
    try {
      const response = await apiClient.getOrders()
      dispatch({ type: 'SET_ORDERS', payload: response.data })
    } catch (error) {
      console.error('Failed to refresh orders:', error)
    }
  }, [apiClient])

  // Position Management
  const getPosition = useCallback((symbol: string) => {
    return state.positions[symbol]
  }, [state.positions])

  const closePosition = useCallback(async (symbol: string, quantity?: number) => {
    await apiClient.closePosition(symbol, quantity)
  }, [apiClient])

  const refreshPositions = useCallback(async () => {
    try {
      const positions = await apiClient.getPositions()
      dispatch({ type: 'SET_POSITIONS', payload: positions })
    } catch (error) {
      console.error('Failed to refresh positions:', error)
    }
  }, [apiClient])

  // Portfolio Management
  const refreshPortfolio = useCallback(async () => {
    try {
      const portfolio = await apiClient.getPortfolio()
      dispatch({ type: 'SET_PORTFOLIO', payload: portfolio })
    } catch (error) {
      console.error('Failed to refresh portfolio:', error)
    }
  }, [apiClient])

  const refreshBalances = useCallback(async () => {
    try {
      const balances = await apiClient.getBalances()
      dispatch({ type: 'SET_BALANCES', payload: balances })
    } catch (error) {
      console.error('Failed to refresh balances:', error)
    }
  }, [apiClient])

  // Risk Management
  const acknowledgeRiskAlert = useCallback((alertId: string) => {
    dispatch({ type: 'ACKNOWLEDGE_RISK_ALERT', payload: alertId })
  }, [])

  const getActiveRiskAlerts = useCallback(() => {
    return state.riskAlerts
  }, [state.riskAlerts])

  // Status checks
  const isConnected = useCallback(() => {
    return state.connectionState === 'connected'
  }, [state.connectionState])

  const isAuthenticated = useCallback(() => {
    return state.authenticated
  }, [state.authenticated])

  const contextValue: TradingContextType = {
    state,
    createOrder,
    cancelOrder,
    modifyOrder,
    refreshOrders,
    getPosition,
    closePosition,
    refreshPositions,
    refreshPortfolio,
    refreshBalances,
    acknowledgeRiskAlert,
    getActiveRiskAlerts,
    connect,
    disconnect,
    authenticate,
    isConnected,
    isAuthenticated
  }

  return (
    <TradingContext.Provider value={contextValue}>
      {children}
    </TradingContext.Provider>
  )
}

// Hook to use Trading Context
export const useTrading = (): TradingContextType => {
  const context = useContext(TradingContext)
  if (!context) {
    throw new Error('useTrading must be used within a TradingProvider')
  }
  return context
}
