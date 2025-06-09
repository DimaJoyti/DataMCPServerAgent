'use client'

/**
 * Market Data Context Provider
 * Manages real-time market data state and WebSocket connections
 */

import React, { createContext, useContext, useReducer, useEffect, useCallback, ReactNode } from 'react'
import { MarketDataWebSocket, Quote, Trade, OrderBookUpdate, Ticker } from '../websocket/MarketDataWebSocket'
import { MarketDataAPI, marketDataAPI } from '../api/MarketDataAPI'
import { MarketData, OrderBookData } from '../api/types'

// State Types
interface MarketDataState {
  quotes: Record<string, Quote>
  trades: Record<string, Trade[]>
  orderBooks: Record<string, OrderBookUpdate>
  tickers: Record<string, Ticker>
  marketData: Record<string, MarketData>
  subscriptions: Set<string>
  connectionState: 'disconnected' | 'connecting' | 'connected' | 'error'
  lastUpdate: number
}

// Action Types
type MarketDataAction =
  | { type: 'SET_CONNECTION_STATE'; payload: MarketDataState['connectionState'] }
  | { type: 'UPDATE_QUOTE'; payload: Quote }
  | { type: 'ADD_TRADE'; payload: Trade }
  | { type: 'UPDATE_ORDER_BOOK'; payload: OrderBookUpdate }
  | { type: 'UPDATE_TICKER'; payload: Ticker }
  | { type: 'SET_MARKET_DATA'; payload: { symbol: string; data: MarketData } }
  | { type: 'ADD_SUBSCRIPTION'; payload: string }
  | { type: 'REMOVE_SUBSCRIPTION'; payload: string }
  | { type: 'CLEAR_DATA' }

// Context Type
interface MarketDataContextType {
  state: MarketDataState
  subscribe: (symbol: string) => void
  unsubscribe: (symbol: string) => void
  getQuote: (symbol: string) => Quote | undefined
  getTrades: (symbol: string) => Trade[]
  getOrderBook: (symbol: string) => OrderBookUpdate | undefined
  getTicker: (symbol: string) => Ticker | undefined
  getMarketData: (symbol: string) => MarketData | undefined
  refreshMarketData: (symbol: string) => Promise<void>
  isSubscribed: (symbol: string) => boolean
  isConnected: () => boolean
}

// Initial State
const initialState: MarketDataState = {
  quotes: {},
  trades: {},
  orderBooks: {},
  tickers: {},
  marketData: {},
  subscriptions: new Set(),
  connectionState: 'disconnected',
  lastUpdate: 0
}

// Reducer
function marketDataReducer(state: MarketDataState, action: MarketDataAction): MarketDataState {
  switch (action.type) {
    case 'SET_CONNECTION_STATE':
      return {
        ...state,
        connectionState: action.payload
      }

    case 'UPDATE_QUOTE':
      return {
        ...state,
        quotes: {
          ...state.quotes,
          [action.payload.symbol]: action.payload
        },
        lastUpdate: Date.now()
      }

    case 'ADD_TRADE':
      const existingTrades = state.trades[action.payload.symbol] || []
      const newTrades = [action.payload, ...existingTrades].slice(0, 100) // Keep last 100 trades
      
      return {
        ...state,
        trades: {
          ...state.trades,
          [action.payload.symbol]: newTrades
        },
        lastUpdate: Date.now()
      }

    case 'UPDATE_ORDER_BOOK':
      return {
        ...state,
        orderBooks: {
          ...state.orderBooks,
          [action.payload.symbol]: action.payload
        },
        lastUpdate: Date.now()
      }

    case 'UPDATE_TICKER':
      return {
        ...state,
        tickers: {
          ...state.tickers,
          [action.payload.symbol]: action.payload
        },
        lastUpdate: Date.now()
      }

    case 'SET_MARKET_DATA':
      return {
        ...state,
        marketData: {
          ...state.marketData,
          [action.payload.symbol]: action.payload.data
        },
        lastUpdate: Date.now()
      }

    case 'ADD_SUBSCRIPTION':
      const newSubscriptions = new Set(state.subscriptions)
      newSubscriptions.add(action.payload)
      return {
        ...state,
        subscriptions: newSubscriptions
      }

    case 'REMOVE_SUBSCRIPTION':
      const updatedSubscriptions = new Set(state.subscriptions)
      updatedSubscriptions.delete(action.payload)
      return {
        ...state,
        subscriptions: updatedSubscriptions
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
const MarketDataContext = createContext<MarketDataContextType | undefined>(undefined)

// Provider Props
interface MarketDataProviderProps {
  children: ReactNode
  wsUrl?: string
  apiUrl?: string
  autoConnect?: boolean
}

// Provider Component
export const MarketDataProvider: React.FC<MarketDataProviderProps> = ({
  children,
  wsUrl = 'ws://localhost:8765/ws/market-data',
  apiUrl = 'http://localhost:8000/api/v1',
  autoConnect = true
}) => {
  const [state, dispatch] = useReducer(marketDataReducer, initialState)
  const [wsClient, setWsClient] = React.useState<MarketDataWebSocket | null>(null)
  const [apiClient] = React.useState(() => new MarketDataAPI(apiUrl))

  // Initialize WebSocket connection
  useEffect(() => {
    if (!autoConnect) return

    const ws = new MarketDataWebSocket({ url: wsUrl })
    setWsClient(ws)

    // Setup event handlers
    ws.addEventListener('stateChange', (state) => {
      dispatch({ type: 'SET_CONNECTION_STATE', payload: state as any })
    })

    ws.addEventListener('error', (error) => {
      console.error('Market data WebSocket error:', error)
      dispatch({ type: 'SET_CONNECTION_STATE', payload: 'error' })
    })

    // Connect
    ws.connect().catch(console.error)

    return () => {
      ws.disconnect()
    }
  }, [wsUrl, autoConnect])

  // Subscribe to symbol
  const subscribe = useCallback((symbol: string) => {
    if (!wsClient || state.subscriptions.has(symbol)) return

    dispatch({ type: 'ADD_SUBSCRIPTION', payload: symbol })

    // Subscribe to all data types for the symbol
    wsClient.subscribeToSymbol(symbol, {
      onQuote: (quote) => {
        dispatch({ type: 'UPDATE_QUOTE', payload: quote })
      },
      onTrade: (trade) => {
        dispatch({ type: 'ADD_TRADE', payload: trade })
      },
      onOrderBook: (orderBook) => {
        dispatch({ type: 'UPDATE_ORDER_BOOK', payload: orderBook })
      },
      onTicker: (ticker) => {
        dispatch({ type: 'UPDATE_TICKER', payload: ticker })
      },
      onError: (error) => {
        console.error(`Market data error for ${symbol}:`, error)
      }
    })

    // Also fetch initial REST data
    refreshMarketData(symbol).catch(console.error)
  }, [wsClient, state.subscriptions])

  // Unsubscribe from symbol
  const unsubscribe = useCallback((symbol: string) => {
    if (!wsClient || !state.subscriptions.has(symbol)) return

    dispatch({ type: 'REMOVE_SUBSCRIPTION', payload: symbol })
    wsClient.unsubscribeFromSymbol(symbol)
  }, [wsClient, state.subscriptions])

  // Refresh market data via REST API
  const refreshMarketData = useCallback(async (symbol: string) => {
    try {
      const data = await apiClient.getMarketData(symbol)
      dispatch({ type: 'SET_MARKET_DATA', payload: { symbol, data } })
    } catch (error) {
      console.error(`Failed to refresh market data for ${symbol}:`, error)
    }
  }, [apiClient])

  // Getter functions
  const getQuote = useCallback((symbol: string) => {
    return state.quotes[symbol]
  }, [state.quotes])

  const getTrades = useCallback((symbol: string) => {
    return state.trades[symbol] || []
  }, [state.trades])

  const getOrderBook = useCallback((symbol: string) => {
    return state.orderBooks[symbol]
  }, [state.orderBooks])

  const getTicker = useCallback((symbol: string) => {
    return state.tickers[symbol]
  }, [state.tickers])

  const getMarketData = useCallback((symbol: string) => {
    return state.marketData[symbol]
  }, [state.marketData])

  const isSubscribed = useCallback((symbol: string) => {
    return state.subscriptions.has(symbol)
  }, [state.subscriptions])

  const isConnected = useCallback(() => {
    return state.connectionState === 'connected'
  }, [state.connectionState])

  const contextValue: MarketDataContextType = {
    state,
    subscribe,
    unsubscribe,
    getQuote,
    getTrades,
    getOrderBook,
    getTicker,
    getMarketData,
    refreshMarketData,
    isSubscribed,
    isConnected
  }

  return (
    <MarketDataContext.Provider value={contextValue}>
      {children}
    </MarketDataContext.Provider>
  )
}

// Hook to use Market Data Context
export const useMarketData = (): MarketDataContextType => {
  const context = useContext(MarketDataContext)
  if (!context) {
    throw new Error('useMarketData must be used within a MarketDataProvider')
  }
  return context
}

// Hook for specific symbol data
export const useSymbolData = (symbol: string) => {
  const { subscribe, unsubscribe, getQuote, getTrades, getOrderBook, getTicker, getMarketData, isSubscribed } = useMarketData()

  useEffect(() => {
    if (symbol && !isSubscribed(symbol)) {
      subscribe(symbol)
    }

    return () => {
      if (symbol && isSubscribed(symbol)) {
        unsubscribe(symbol)
      }
    }
  }, [symbol, subscribe, unsubscribe, isSubscribed])

  return {
    quote: getQuote(symbol),
    trades: getTrades(symbol),
    orderBook: getOrderBook(symbol),
    ticker: getTicker(symbol),
    marketData: getMarketData(symbol),
    isSubscribed: isSubscribed(symbol)
  }
}
