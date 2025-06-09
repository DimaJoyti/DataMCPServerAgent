/**
 * Trading Hooks
 * Custom hooks for trading operations and state management
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import { useTrading as useTradingContext } from '../contexts/TradingContext'
import { tradingAPI } from '../api/TradingAPI'
import { 
  CreateOrderRequest, 
  Order, 
  Position, 
  Portfolio, 
  Balance, 
  RiskMetrics, 
  PerformanceMetrics,
  OrderFilter,
  PaginationParams
} from '../api/types'

// Hook for order management
export const useOrders = (filter?: OrderFilter, pagination?: PaginationParams) => {
  const { state, refreshOrders } = useTradingContext()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const orders = useMemo(() => {
    let filteredOrders = Object.values(state.orders)

    if (filter) {
      if (filter.symbol) {
        filteredOrders = filteredOrders.filter(order => order.symbol === filter.symbol)
      }
      if (filter.side) {
        filteredOrders = filteredOrders.filter(order => order.side === filter.side)
      }
      if (filter.status) {
        filteredOrders = filteredOrders.filter(order => order.status === filter.status)
      }
    }

    return filteredOrders.sort((a, b) => 
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    )
  }, [state.orders, filter])

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      await refreshOrders()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh orders')
    } finally {
      setLoading(false)
    }
  }, [refreshOrders])

  return {
    orders,
    loading,
    error,
    refresh,
    openOrders: orders.filter(order => ['pending', 'partially_filled'].includes(order.status)),
    filledOrders: orders.filter(order => order.status === 'filled'),
    cancelledOrders: orders.filter(order => order.status === 'cancelled')
  }
}

// Hook for creating orders
export const useCreateOrder = () => {
  const { createOrder } = useTradingContext()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastOrder, setLastOrder] = useState<Order | null>(null)

  const submitOrder = useCallback(async (orderRequest: CreateOrderRequest) => {
    setLoading(true)
    setError(null)
    
    try {
      const order = await createOrder(orderRequest)
      setLastOrder(order)
      return order
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create order'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [createOrder])

  return {
    submitOrder,
    loading,
    error,
    lastOrder
  }
}

// Hook for position management
export const usePositions = () => {
  const { state, refreshPositions, getPosition, closePosition } = useTradingContext()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const positions = useMemo(() => {
    return Object.values(state.positions).filter(position => position.quantity !== 0)
  }, [state.positions])

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      await refreshPositions()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh positions')
    } finally {
      setLoading(false)
    }
  }, [refreshPositions])

  const close = useCallback(async (symbol: string, quantity?: number) => {
    setLoading(true)
    setError(null)
    
    try {
      await closePosition(symbol, quantity)
      await refreshPositions()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to close position')
      throw err
    } finally {
      setLoading(false)
    }
  }, [closePosition, refreshPositions])

  return {
    positions,
    loading,
    error,
    refresh,
    closePosition: close,
    getPosition,
    longPositions: positions.filter(pos => pos.side === 'long'),
    shortPositions: positions.filter(pos => pos.side === 'short'),
    totalUnrealizedPnL: positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0),
    totalRealizedPnL: positions.reduce((sum, pos) => sum + pos.realizedPnL, 0)
  }
}

// Hook for portfolio data
export const usePortfolio = () => {
  const { state, refreshPortfolio } = useTradingContext()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      await refreshPortfolio()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh portfolio')
    } finally {
      setLoading(false)
    }
  }, [refreshPortfolio])

  return {
    portfolio: state.portfolio,
    loading,
    error,
    refresh
  }
}

// Hook for account balances
export const useBalances = () => {
  const { state, refreshBalances } = useTradingContext()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const balances = useMemo(() => {
    return Object.values(state.balances)
  }, [state.balances])

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      await refreshBalances()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh balances')
    } finally {
      setLoading(false)
    }
  }, [refreshBalances])

  const getBalance = useCallback((currency: string) => {
    return state.balances[currency]
  }, [state.balances])

  return {
    balances,
    loading,
    error,
    refresh,
    getBalance,
    totalValue: balances.reduce((sum, balance) => sum + balance.total, 0)
  }
}

// Hook for risk metrics
export const useRiskMetrics = () => {
  const [metrics, setMetrics] = useState<RiskMetrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchMetrics = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      const riskMetrics = await tradingAPI.getRiskMetrics()
      setMetrics(riskMetrics)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch risk metrics')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchMetrics()
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchMetrics, 30000)
    return () => clearInterval(interval)
  }, [fetchMetrics])

  return {
    metrics,
    loading,
    error,
    refresh: fetchMetrics
  }
}

// Hook for performance metrics
export const usePerformanceMetrics = (timeframe?: string) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchMetrics = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      const performanceMetrics = await tradingAPI.getPerformanceMetrics(timeframe)
      setMetrics(performanceMetrics)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch performance metrics')
    } finally {
      setLoading(false)
    }
  }, [timeframe])

  useEffect(() => {
    fetchMetrics()
  }, [fetchMetrics])

  return {
    metrics,
    loading,
    error,
    refresh: fetchMetrics
  }
}

// Hook for risk alerts
export const useRiskAlerts = () => {
  const { state, acknowledgeRiskAlert, getActiveRiskAlerts } = useTradingContext()

  const alerts = useMemo(() => {
    return getActiveRiskAlerts()
  }, [getActiveRiskAlerts])

  const acknowledge = useCallback((alertId: string) => {
    acknowledgeRiskAlert(alertId)
  }, [acknowledgeRiskAlert])

  return {
    alerts,
    acknowledge,
    criticalAlerts: alerts.filter(alert => alert.severity === 'critical'),
    highAlerts: alerts.filter(alert => alert.severity === 'high'),
    mediumAlerts: alerts.filter(alert => alert.severity === 'medium'),
    lowAlerts: alerts.filter(alert => alert.severity === 'low')
  }
}

// Hook for trading connection status
export const useTradingConnection = () => {
  const { state, connect, disconnect, authenticate, isConnected, isAuthenticated } = useTradingContext()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const connectAndAuth = useCallback(async (apiKey: string, apiSecret: string) => {
    setLoading(true)
    setError(null)
    
    try {
      await connect()
      await authenticate(apiKey, apiSecret)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect and authenticate')
      throw err
    } finally {
      setLoading(false)
    }
  }, [connect, authenticate])

  return {
    connectionState: state.connectionState,
    isConnected: isConnected(),
    isAuthenticated: isAuthenticated(),
    loading,
    error,
    connect: connectAndAuth,
    disconnect
  }
}

// Hook for order execution tracking
export const useOrderExecution = (orderId?: string) => {
  const { state } = useTradingContext()

  const order = orderId ? state.orders[orderId] : null
  const executions = useMemo(() => {
    return state.recentTrades.filter(trade => trade.orderId === orderId)
  }, [state.recentTrades, orderId])

  return {
    order,
    executions,
    totalExecuted: executions.reduce((sum, trade) => sum + trade.quantity, 0),
    averagePrice: executions.length > 0 
      ? executions.reduce((sum, trade) => sum + (trade.price * trade.quantity), 0) / 
        executions.reduce((sum, trade) => sum + trade.quantity, 0)
      : 0,
    totalCommission: executions.reduce((sum, trade) => sum + trade.commission, 0)
  }
}

// Hook for P&L tracking
export const usePnLTracking = () => {
  const { state } = useTradingContext()

  const pnlData = useMemo(() => {
    const positions = Object.values(state.positions)
    const totalUnrealizedPnL = positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0)
    const totalRealizedPnL = positions.reduce((sum, pos) => sum + pos.realizedPnL, 0)
    const totalPnL = totalUnrealizedPnL + totalRealizedPnL

    return {
      totalPnL,
      totalUnrealizedPnL,
      totalRealizedPnL,
      dayPnL: state.portfolio?.dayPnL || 0,
      dayPnLPercent: state.portfolio?.dayPnLPercent || 0,
      winningPositions: positions.filter(pos => pos.unrealizedPnL > 0).length,
      losingPositions: positions.filter(pos => pos.unrealizedPnL < 0).length
    }
  }, [state.positions, state.portfolio])

  return pnlData
}
