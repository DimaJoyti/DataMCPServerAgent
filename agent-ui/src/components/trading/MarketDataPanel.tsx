'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  RefreshCw,
  Wifi,
  WifiOff
} from 'lucide-react'
import { useSymbol, useOrderBook, useMarketDataConnection } from '@/services/hooks/useMarketData'

interface MarketData {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  high24h: number
  low24h: number
  bid: number
  ask: number
  spread: number
  lastUpdate: Date
}

interface OrderBookLevel {
  price: number
  size: number
  total: number
}

export const MarketDataPanel: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USD')

  // Use real-time hooks
  const symbolData = useSymbol(selectedSymbol)
  const { orderBook } = useOrderBook(selectedSymbol)
  const { isConnected, connectionState } = useMarketDataConnection()

  // Fallback data for when real-time data is not available
  const marketData = {
    symbol: selectedSymbol,
    price: symbolData.price || 43250.75,
    change: symbolData.change || 1250.25,
    changePercent: symbolData.changePercent || 2.98,
    volume: symbolData.volume || 125000000,
    high24h: symbolData.marketData?.high24h || 44100.00,
    low24h: symbolData.marketData?.low24h || 41800.50,
    bid: symbolData.quote?.bid || 43248.50,
    ask: symbolData.quote?.ask || 43252.25,
    spread: symbolData.quote ? (symbolData.quote.ask - symbolData.quote.bid) : 3.75,
    lastUpdate: new Date()
  }

  // Use real-time order book data or fallback
  const bids = orderBook?.bids?.map((level, index) => ({
    price: level.price,
    size: level.size,
    total: orderBook.bids.slice(0, index + 1).reduce((sum, l) => sum + l.size, 0)
  })) || [
    { price: 43248.50, size: 2.5, total: 2.5 },
    { price: 43247.25, size: 1.8, total: 4.3 },
    { price: 43246.00, size: 3.2, total: 7.5 },
    { price: 43244.75, size: 0.9, total: 8.4 },
    { price: 43243.50, size: 2.1, total: 10.5 }
  ]

  const asks = orderBook?.asks?.map((level, index) => ({
    price: level.price,
    size: level.size,
    total: orderBook.asks.slice(0, index + 1).reduce((sum, l) => sum + l.size, 0)
  })) || [
    { price: 43252.25, size: 1.9, total: 1.9 },
    { price: 43253.50, size: 2.7, total: 4.6 },
    { price: 43254.75, size: 1.5, total: 6.1 },
    { price: 43256.00, size: 3.1, total: 9.2 },
    { price: 43257.25, size: 0.8, total: 10.0 }
  ]

  const symbols = ['BTC/USD', 'ETH/USD', 'AAPL', 'TSLA', 'SPY', 'QQQ', 'EUR/USD', 'GBP/USD']

  const handleSymbolChange = (symbol: string) => {
    setSelectedSymbol(symbol)
    // In a real app, this would trigger a new data subscription
    setMarketData(prev => ({
      ...prev,
      symbol,
      price: 100 + Math.random() * 1000,
      change: (Math.random() - 0.5) * 50,
      changePercent: (Math.random() - 0.5) * 10
    }))
  }

  const formatPrice = (price: number) => {
    return price.toFixed(2)
  }

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`
    }
    if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`
    }
    return volume.toString()
  }

  return (
    <Card className="h-full bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl font-bold">Market Data</CardTitle>
            <CardDescription>Real-time market information and order book</CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1">
              {isConnected ? (
                <Wifi className="h-4 w-4 text-green-500" />
              ) : (
                <WifiOff className="h-4 w-4 text-red-500" />
              )}
              <span className="text-xs text-muted-foreground">
                {isConnected ? 'Live' : 'Disconnected'}
              </span>
            </div>
            <Button variant="outline" size="sm">
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Symbol Selector */}
        <div className="flex items-center space-x-4">
          <Select value={selectedSymbol} onValueChange={handleSymbolChange}>
            <SelectTrigger className="w-40">
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
          <Badge variant="outline" className="text-xs">
            Last: {marketData.lastUpdate.toLocaleTimeString()}
          </Badge>
        </div>

        {/* Price Information */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className="text-2xl font-bold">${formatPrice(marketData.price)}</span>
              <div className={`flex items-center space-x-1 ${
                marketData.change >= 0 ? 'text-green-500' : 'text-red-500'
              }`}>
                {marketData.change >= 0 ? (
                  <TrendingUp className="h-4 w-4" />
                ) : (
                  <TrendingDown className="h-4 w-4" />
                )}
                <span className="font-medium">
                  {marketData.change >= 0 ? '+' : ''}${formatPrice(marketData.change)}
                </span>
                <span className="text-sm">
                  ({marketData.changePercent >= 0 ? '+' : ''}{marketData.changePercent.toFixed(2)}%)
                </span>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-muted-foreground">Bid:</span>
                <span className="ml-2 font-medium text-green-500">${formatPrice(marketData.bid)}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Ask:</span>
                <span className="ml-2 font-medium text-red-500">${formatPrice(marketData.ask)}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Spread:</span>
                <span className="ml-2 font-medium">${formatPrice(marketData.spread)}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Volume:</span>
                <span className="ml-2 font-medium">{formatVolume(marketData.volume)}</span>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">24h High:</span>
                <span className="font-medium">${formatPrice(marketData.high24h)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">24h Low:</span>
                <span className="font-medium">${formatPrice(marketData.low24h)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Order Book */}
        <div className="space-y-4">
          <h4 className="font-semibold flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Order Book</span>
          </h4>
          
          <div className="grid grid-cols-2 gap-4">
            {/* Bids */}
            <div>
              <div className="text-xs text-muted-foreground mb-2 grid grid-cols-3 gap-2">
                <span>Price</span>
                <span className="text-right">Size</span>
                <span className="text-right">Total</span>
              </div>
              <div className="space-y-1">
                {bids.map((bid, index) => (
                  <div key={index} className="text-xs grid grid-cols-3 gap-2 py-1 hover:bg-green-50 dark:hover:bg-green-900/20 rounded">
                    <span className="text-green-500 font-medium">${formatPrice(bid.price)}</span>
                    <span className="text-right">{bid.size.toFixed(3)}</span>
                    <span className="text-right text-muted-foreground">{bid.total.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Asks */}
            <div>
              <div className="text-xs text-muted-foreground mb-2 grid grid-cols-3 gap-2">
                <span>Price</span>
                <span className="text-right">Size</span>
                <span className="text-right">Total</span>
              </div>
              <div className="space-y-1">
                {asks.map((ask, index) => (
                  <div key={index} className="text-xs grid grid-cols-3 gap-2 py-1 hover:bg-red-50 dark:hover:bg-red-900/20 rounded">
                    <span className="text-red-500 font-medium">${formatPrice(ask.price)}</span>
                    <span className="text-right">{ask.size.toFixed(3)}</span>
                    <span className="text-right text-muted-foreground">{ask.total.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
