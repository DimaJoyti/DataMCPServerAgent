/**
 * Market Data Hooks
 * Custom hooks for consuming market data in components
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import { useMarketData as useMarketDataContext, useSymbolData } from '../contexts/MarketDataContext'
import { marketDataAPI } from '../api/MarketDataAPI'
import { MarketData, OrderBookData, OHLCV, HistoricalDataRequest } from '../api/types'

// Hook for real-time symbol data
export const useSymbol = (symbol: string) => {
  const symbolData = useSymbolData(symbol)
  
  return {
    ...symbolData,
    price: symbolData.quote?.bid || symbolData.ticker?.price || symbolData.marketData?.price || 0,
    change: symbolData.ticker?.change || symbolData.marketData?.change || 0,
    changePercent: symbolData.ticker?.changePercent || symbolData.marketData?.changePercent || 0,
    volume: symbolData.ticker?.volume || symbolData.marketData?.volume || 0,
    isLoading: !symbolData.quote && !symbolData.ticker && !symbolData.marketData
  }
}

// Hook for multiple symbols
export const useMultipleSymbols = (symbols: string[]) => {
  const { subscribe, unsubscribe, getQuote, getTicker, getMarketData, isSubscribed } = useMarketDataContext()
  
  useEffect(() => {
    symbols.forEach(symbol => {
      if (!isSubscribed(symbol)) {
        subscribe(symbol)
      }
    })

    return () => {
      symbols.forEach(symbol => {
        if (isSubscribed(symbol)) {
          unsubscribe(symbol)
        }
      })
    }
  }, [symbols, subscribe, unsubscribe, isSubscribed])

  return useMemo(() => {
    return symbols.map(symbol => {
      const quote = getQuote(symbol)
      const ticker = getTicker(symbol)
      const marketData = getMarketData(symbol)
      
      return {
        symbol,
        quote,
        ticker,
        marketData,
        price: quote?.bid || ticker?.price || marketData?.price || 0,
        change: ticker?.change || marketData?.change || 0,
        changePercent: ticker?.changePercent || marketData?.changePercent || 0,
        volume: ticker?.volume || marketData?.volume || 0,
        isLoading: !quote && !ticker && !marketData
      }
    })
  }, [symbols, getQuote, getTicker, getMarketData])
}

// Hook for order book data
export const useOrderBook = (symbol: string, depth: number = 20) => {
  const { getOrderBook, isSubscribed, subscribe } = useMarketDataContext()
  const [restOrderBook, setRestOrderBook] = useState<OrderBookData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Subscribe to real-time order book
  useEffect(() => {
    if (!isSubscribed(symbol)) {
      subscribe(symbol)
    }
  }, [symbol, isSubscribed, subscribe])

  // Fetch initial order book via REST API
  useEffect(() => {
    const fetchOrderBook = async () => {
      setLoading(true)
      setError(null)
      
      try {
        const data = await marketDataAPI.getOrderBook(symbol, depth)
        setRestOrderBook(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch order book')
      } finally {
        setLoading(false)
      }
    }

    fetchOrderBook()
  }, [symbol, depth])

  const wsOrderBook = getOrderBook(symbol)
  const orderBook = wsOrderBook || restOrderBook

  return {
    orderBook,
    loading,
    error,
    spread: orderBook ? (orderBook.asks[0]?.price || 0) - (orderBook.bids[0]?.price || 0) : 0,
    midPrice: orderBook ? ((orderBook.asks[0]?.price || 0) + (orderBook.bids[0]?.price || 0)) / 2 : 0
  }
}

// Hook for historical data
export const useHistoricalData = (request: HistoricalDataRequest) => {
  const [data, setData] = useState<OHLCV[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      const historicalData = await marketDataAPI.getHistoricalData(request)
      setData(historicalData)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch historical data')
    } finally {
      setLoading(false)
    }
  }, [request])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  return {
    data,
    loading,
    error,
    refetch: fetchData
  }
}

// Hook for market statistics
export const useMarketStats = () => {
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchStats = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      const marketStats = await marketDataAPI.getMarketStats()
      setStats(marketStats)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch market stats')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStats()
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchStats, 30000)
    return () => clearInterval(interval)
  }, [fetchStats])

  return {
    stats,
    loading,
    error,
    refetch: fetchStats
  }
}

// Hook for top movers
export const useTopMovers = (type: 'gainers' | 'losers' = 'gainers', limit: number = 10) => {
  const [movers, setMovers] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchMovers = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      const topMovers = await marketDataAPI.getTopMovers(type, limit)
      setMovers(topMovers)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch top movers')
    } finally {
      setLoading(false)
    }
  }, [type, limit])

  useEffect(() => {
    fetchMovers()
    
    // Refresh every 60 seconds
    const interval = setInterval(fetchMovers, 60000)
    return () => clearInterval(interval)
  }, [fetchMovers])

  return {
    movers,
    loading,
    error,
    refetch: fetchMovers
  }
}

// Hook for symbol search
export const useSymbolSearch = () => {
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const search = useCallback(async (query: string, limit: number = 20) => {
    if (!query.trim()) {
      setResults([])
      return
    }

    setLoading(true)
    setError(null)
    
    try {
      const searchResults = await marketDataAPI.searchSymbols(query, limit)
      setResults(searchResults)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to search symbols')
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    results,
    loading,
    error,
    search
  }
}

// Hook for market hours
export const useMarketHours = (exchange?: string) => {
  const [hours, setHours] = useState<any>(null)
  const [isOpen, setIsOpen] = useState<boolean>(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchMarketHours = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      const [marketHours, marketOpen] = await Promise.all([
        marketDataAPI.getMarketHours(exchange),
        marketDataAPI.isMarketOpen(exchange)
      ])
      
      setHours(marketHours)
      setIsOpen(marketOpen)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch market hours')
    } finally {
      setLoading(false)
    }
  }, [exchange])

  useEffect(() => {
    fetchMarketHours()
    
    // Refresh every 5 minutes
    const interval = setInterval(fetchMarketHours, 300000)
    return () => clearInterval(interval)
  }, [fetchMarketHours])

  return {
    hours,
    isOpen,
    loading,
    error,
    refetch: fetchMarketHours
  }
}

// Hook for price alerts (client-side)
export const usePriceAlert = (symbol: string, targetPrice: number, condition: 'above' | 'below') => {
  const { price } = useSymbol(symbol)
  const [triggered, setTriggered] = useState(false)
  const [lastTriggerTime, setLastTriggerTime] = useState<Date | null>(null)

  useEffect(() => {
    if (!price || triggered) return

    const shouldTrigger = condition === 'above' ? price >= targetPrice : price <= targetPrice

    if (shouldTrigger) {
      setTriggered(true)
      setLastTriggerTime(new Date())
      
      // You could add notification logic here
      console.log(`Price alert triggered for ${symbol}: ${price} is ${condition} ${targetPrice}`)
    }
  }, [price, targetPrice, condition, triggered, symbol])

  const reset = useCallback(() => {
    setTriggered(false)
    setLastTriggerTime(null)
  }, [])

  return {
    triggered,
    lastTriggerTime,
    currentPrice: price,
    reset
  }
}

// Hook for connection status
export const useMarketDataConnection = () => {
  const { state, isConnected } = useMarketDataContext()
  
  return {
    connectionState: state.connectionState,
    isConnected: isConnected(),
    lastUpdate: state.lastUpdate,
    subscriptionCount: state.subscriptions.size
  }
}
