'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Activity,
  Layers,
  RefreshCw
} from 'lucide-react'

interface OrderBookLevel {
  price: number
  size: number
  total: number
  orders: number
}

interface OrderBookData {
  symbol: string
  bids: OrderBookLevel[]
  asks: OrderBookLevel[]
  spread: number
  spreadPercent: number
  lastUpdate: Date
}

export const OrderBook: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USD')
  const [precision, setPrecision] = useState('0.01')
  const [orderBookData, setOrderBookData] = useState<OrderBookData>({
    symbol: 'BTC/USD',
    bids: [
      { price: 43248.50, size: 2.5, total: 2.5, orders: 3 },
      { price: 43247.25, size: 1.8, total: 4.3, orders: 2 },
      { price: 43246.00, size: 3.2, total: 7.5, orders: 5 },
      { price: 43244.75, size: 0.9, total: 8.4, orders: 1 },
      { price: 43243.50, size: 2.1, total: 10.5, orders: 4 },
      { price: 43242.25, size: 1.5, total: 12.0, orders: 2 },
      { price: 43241.00, size: 2.8, total: 14.8, orders: 3 },
      { price: 43239.75, size: 1.2, total: 16.0, orders: 2 },
      { price: 43238.50, size: 3.5, total: 19.5, orders: 6 },
      { price: 43237.25, size: 0.7, total: 20.2, orders: 1 }
    ],
    asks: [
      { price: 43252.25, size: 1.9, total: 1.9, orders: 2 },
      { price: 43253.50, size: 2.7, total: 4.6, orders: 4 },
      { price: 43254.75, size: 1.5, total: 6.1, orders: 2 },
      { price: 43256.00, size: 3.1, total: 9.2, orders: 5 },
      { price: 43257.25, size: 0.8, total: 10.0, orders: 1 },
      { price: 43258.50, size: 2.4, total: 12.4, orders: 3 },
      { price: 43259.75, size: 1.6, total: 14.0, orders: 2 },
      { price: 43261.00, size: 2.9, total: 16.9, orders: 4 },
      { price: 43262.25, size: 1.1, total: 18.0, orders: 1 },
      { price: 43263.50, size: 3.3, total: 21.3, orders: 6 }
    ],
    spread: 3.75,
    spreadPercent: 0.0087,
    lastUpdate: new Date()
  })

  const symbols = ['BTC/USD', 'ETH/USD', 'AAPL', 'TSLA', 'SPY', 'QQQ']
  const precisionOptions = ['0.01', '0.1', '1', '10']

  useEffect(() => {
    // Simulate real-time order book updates
    const interval = setInterval(() => {
      setOrderBookData(prev => {
        const newBids = prev.bids.map(level => ({
          ...level,
          size: Math.max(0.1, level.size + (Math.random() - 0.5) * 0.5),
          orders: Math.max(1, level.orders + Math.floor((Math.random() - 0.5) * 2))
        }))

        const newAsks = prev.asks.map(level => ({
          ...level,
          size: Math.max(0.1, level.size + (Math.random() - 0.5) * 0.5),
          orders: Math.max(1, level.orders + Math.floor((Math.random() - 0.5) * 2))
        }))

        // Recalculate totals
        let bidTotal = 0
        const bidsWithTotals = newBids.map(level => {
          bidTotal += level.size
          return { ...level, total: bidTotal }
        })

        let askTotal = 0
        const asksWithTotals = newAsks.map(level => {
          askTotal += level.size
          return { ...level, total: askTotal }
        })

        const bestBid = bidsWithTotals[0]?.price || 0
        const bestAsk = asksWithTotals[0]?.price || 0
        const spread = bestAsk - bestBid
        const spreadPercent = spread / bestBid * 100

        return {
          ...prev,
          bids: bidsWithTotals,
          asks: asksWithTotals,
          spread,
          spreadPercent,
          lastUpdate: new Date()
        }
      })
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  const handleSymbolChange = (symbol: string) => {
    setSelectedSymbol(symbol)
    // In a real app, this would trigger a new order book subscription
  }

  const formatPrice = (price: number) => {
    const prec = parseFloat(precision)
    return price.toFixed(prec >= 1 ? 0 : prec === 0.1 ? 1 : 2)
  }

  const formatSize = (size: number) => {
    return size.toFixed(3)
  }

  const getMaxSize = () => {
    const maxBidSize = Math.max(...orderBookData.bids.map(b => b.size))
    const maxAskSize = Math.max(...orderBookData.asks.map(a => a.size))
    return Math.max(maxBidSize, maxAskSize)
  }

  const getSizeBarWidth = (size: number) => {
    const maxSize = getMaxSize()
    return (size / maxSize) * 100
  }

  return (
    <Card className="h-full bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl font-bold flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Order Book</span>
            </CardTitle>
            <CardDescription>Real-time market depth and liquidity</CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm">
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Controls */}
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <Select value={selectedSymbol} onValueChange={handleSymbolChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {symbols.map(symbol => (
                  <SelectItem key={symbol} value={symbol}>
                    {symbol}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Select value={precision} onValueChange={setPrecision}>
              <SelectTrigger className="w-24">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {precisionOptions.map(prec => (
                  <SelectItem key={prec} value={prec}>
                    {prec}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Spread Information */}
        <div className="flex items-center justify-between p-3 bg-muted/50 rounded-md">
          <div className="flex items-center space-x-2">
            <Layers className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Spread:</span>
          </div>
          <div className="flex items-center space-x-4">
            <span className="font-medium">${orderBookData.spread.toFixed(2)}</span>
            <Badge variant="outline" className="text-xs">
              {orderBookData.spreadPercent.toFixed(4)}%
            </Badge>
            <span className="text-xs text-muted-foreground">
              {orderBookData.lastUpdate.toLocaleTimeString()}
            </span>
          </div>
        </div>

        {/* Order Book */}
        <div className="space-y-2">
          {/* Header */}
          <div className="grid grid-cols-4 gap-2 text-xs text-muted-foreground font-medium border-b pb-2">
            <span>Price</span>
            <span className="text-right">Size</span>
            <span className="text-right">Total</span>
            <span className="text-right">Orders</span>
          </div>

          {/* Asks (Sell Orders) */}
          <div className="space-y-1">
            {orderBookData.asks.slice().reverse().map((ask, index) => (
              <div key={`ask-${index}`} className="relative">
                <div 
                  className="absolute inset-y-0 right-0 bg-red-100 dark:bg-red-900/20 rounded"
                  style={{ width: `${getSizeBarWidth(ask.size)}%` }}
                />
                <div className="relative grid grid-cols-4 gap-2 text-xs py-1 px-2 hover:bg-red-50 dark:hover:bg-red-900/10 rounded cursor-pointer">
                  <span className="text-red-500 font-medium">${formatPrice(ask.price)}</span>
                  <span className="text-right">{formatSize(ask.size)}</span>
                  <span className="text-right text-muted-foreground">{formatSize(ask.total)}</span>
                  <span className="text-right text-muted-foreground">{ask.orders}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Spread Indicator */}
          <div className="flex items-center justify-center py-2 border-y">
            <div className="flex items-center space-x-2 text-sm">
              <TrendingUp className="h-4 w-4 text-red-500" />
              <span className="font-medium">${orderBookData.spread.toFixed(2)}</span>
              <TrendingDown className="h-4 w-4 text-green-500" />
            </div>
          </div>

          {/* Bids (Buy Orders) */}
          <div className="space-y-1">
            {orderBookData.bids.map((bid, index) => (
              <div key={`bid-${index}`} className="relative">
                <div 
                  className="absolute inset-y-0 right-0 bg-green-100 dark:bg-green-900/20 rounded"
                  style={{ width: `${getSizeBarWidth(bid.size)}%` }}
                />
                <div className="relative grid grid-cols-4 gap-2 text-xs py-1 px-2 hover:bg-green-50 dark:hover:bg-green-900/10 rounded cursor-pointer">
                  <span className="text-green-500 font-medium">${formatPrice(bid.price)}</span>
                  <span className="text-right">{formatSize(bid.size)}</span>
                  <span className="text-right text-muted-foreground">{formatSize(bid.total)}</span>
                  <span className="text-right text-muted-foreground">{bid.orders}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 gap-4 pt-4 border-t">
          <div className="text-center">
            <div className="text-sm text-muted-foreground">Total Bid Volume</div>
            <div className="font-medium text-green-500">
              {formatSize(orderBookData.bids.reduce((sum, bid) => sum + bid.size, 0))}
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-muted-foreground">Total Ask Volume</div>
            <div className="font-medium text-red-500">
              {formatSize(orderBookData.asks.reduce((sum, ask) => sum + ask.size, 0))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
