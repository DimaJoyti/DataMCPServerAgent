'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import {
  PieChart,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  Shield,
  AlertTriangle,
  X,
  Plus
} from 'lucide-react'

interface Position {
  id: string
  symbol: string
  side: 'long' | 'short'
  quantity: number
  entryPrice: number
  currentPrice: number
  unrealizedPnL: number
  unrealizedPnLPercent: number
  realizedPnL: number
  totalValue: number
  stopLoss?: number
  takeProfit?: number
  riskLevel: 'low' | 'medium' | 'high'
  openDate: Date
}

interface PortfolioSummary {
  totalValue: number
  totalPnL: number
  totalPnLPercent: number
  dayPnL: number
  dayPnLPercent: number
  marginUsed: number
  marginAvailable: number
  buyingPower: number
}

export const PositionManager: React.FC = () => {
  const [positions, setPositions] = useState<Position[]>([
    {
      id: 'POS-001',
      symbol: 'BTC/USD',
      side: 'long',
      quantity: 2.5,
      entryPrice: 42800.00,
      currentPrice: 43250.75,
      unrealizedPnL: 1126.88,
      unrealizedPnLPercent: 1.05,
      realizedPnL: 0,
      totalValue: 108126.88,
      stopLoss: 41500.00,
      takeProfit: 45000.00,
      riskLevel: 'medium',
      openDate: new Date(Date.now() - 86400000 * 2)
    },
    {
      id: 'POS-002',
      symbol: 'ETH/USD',
      side: 'long',
      quantity: 10.0,
      entryPrice: 2420.50,
      currentPrice: 2445.25,
      unrealizedPnL: 247.50,
      unrealizedPnLPercent: 1.02,
      realizedPnL: 150.00,
      totalValue: 24452.50,
      riskLevel: 'low',
      openDate: new Date(Date.now() - 86400000 * 1)
    },
    {
      id: 'POS-003',
      symbol: 'AAPL',
      side: 'long',
      quantity: 500,
      entryPrice: 185.20,
      currentPrice: 182.75,
      unrealizedPnL: -1225.00,
      unrealizedPnLPercent: -1.32,
      realizedPnL: 0,
      totalValue: 91375.00,
      stopLoss: 180.00,
      riskLevel: 'high',
      openDate: new Date(Date.now() - 86400000 * 3)
    }
  ])

  const [portfolioSummary, setPortfolioSummary] = useState<PortfolioSummary>({
    totalValue: 223954.38,
    totalPnL: 149.38,
    totalPnLPercent: 0.067,
    dayPnL: 1247.25,
    dayPnLPercent: 0.56,
    marginUsed: 45000.00,
    marginAvailable: 155000.00,
    buyingPower: 400000.00
  })

  useEffect(() => {
    // Simulate real-time position updates
    const interval = setInterval(() => {
      setPositions(prev => prev.map(position => {
        const priceChange = (Math.random() - 0.5) * position.currentPrice * 0.002
        const newPrice = position.currentPrice + priceChange
        const unrealizedPnL = (newPrice - position.entryPrice) * position.quantity * (position.side === 'long' ? 1 : -1)
        const unrealizedPnLPercent = (unrealizedPnL / (position.entryPrice * position.quantity)) * 100

        return {
          ...position,
          currentPrice: newPrice,
          unrealizedPnL,
          unrealizedPnLPercent,
          totalValue: newPrice * position.quantity
        }
      }))
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    // Update portfolio summary when positions change
    const totalValue = positions.reduce((sum, pos) => sum + pos.totalValue, 0)
    const totalPnL = positions.reduce((sum, pos) => sum + pos.unrealizedPnL + pos.realizedPnL, 0)
    const totalPnLPercent = (totalPnL / (totalValue - totalPnL)) * 100

    setPortfolioSummary(prev => ({
      ...prev,
      totalValue,
      totalPnL,
      totalPnLPercent
    }))
  }, [positions])

  const handleClosePosition = (positionId: string) => {
    setPositions(prev => prev.filter(pos => pos.id !== positionId))
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-500'
      case 'medium': return 'text-yellow-500'
      case 'high': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getRiskBadgeVariant = (level: string) => {
    switch (level) {
      case 'low': return 'default'
      case 'medium': return 'secondary'
      case 'high': return 'destructive'
      default: return 'outline'
    }
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(amount)
  }

  const formatPercent = (percent: number) => {
    return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`
  }

  return (
    <div className="h-full space-y-6 p-6">
      {/* Portfolio Summary */}
      <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
        <CardHeader>
          <CardTitle className="text-xl font-bold flex items-center space-x-2">
            <PieChart className="h-5 w-5" />
            <span>Portfolio Summary</span>
          </CardTitle>
          <CardDescription>Overall portfolio performance and metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold">{formatCurrency(portfolioSummary.totalValue)}</div>
              <div className="text-sm text-muted-foreground">Total Value</div>
            </div>
            <div className="text-center">
              <div className={`text-2xl font-bold ${portfolioSummary.totalPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {formatCurrency(portfolioSummary.totalPnL)}
              </div>
              <div className="text-sm text-muted-foreground">
                Total P&L ({formatPercent(portfolioSummary.totalPnLPercent)})
              </div>
            </div>
            <div className="text-center">
              <div className={`text-2xl font-bold ${portfolioSummary.dayPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {formatCurrency(portfolioSummary.dayPnL)}
              </div>
              <div className="text-sm text-muted-foreground">
                Day P&L ({formatPercent(portfolioSummary.dayPnLPercent)})
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">{formatCurrency(portfolioSummary.buyingPower)}</div>
              <div className="text-sm text-muted-foreground">Buying Power</div>
            </div>
          </div>

          <div className="mt-6">
            <div className="flex justify-between text-sm mb-2">
              <span>Margin Used</span>
              <span>{formatCurrency(portfolioSummary.marginUsed)} / {formatCurrency(portfolioSummary.marginUsed + portfolioSummary.marginAvailable)}</span>
            </div>
            <Progress 
              value={(portfolioSummary.marginUsed / (portfolioSummary.marginUsed + portfolioSummary.marginAvailable)) * 100} 
              className="h-2"
            />
          </div>
        </CardContent>
      </Card>

      {/* Positions */}
      <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-xl font-bold">Open Positions</CardTitle>
              <CardDescription>Active trading positions and their performance</CardDescription>
            </div>
            <Button variant="outline" size="sm">
              <Plus className="h-4 w-4 mr-2" />
              New Position
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {positions.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No open positions</p>
                <p className="text-sm">Start trading to see your positions here</p>
              </div>
            ) : (
              positions.map((position) => (
                <div key={position.id} className="border border-border rounded-lg p-4 hover:shadow-md transition-shadow">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="font-bold text-lg">{position.symbol}</span>
                          <Badge variant={position.side === 'long' ? 'default' : 'destructive'}>
                            {position.side.toUpperCase()}
                          </Badge>
                          <Badge variant={getRiskBadgeVariant(position.riskLevel)}>
                            {position.riskLevel.toUpperCase()}
                          </Badge>
                        </div>
                        <div className="text-sm text-muted-foreground">
                          Opened {position.openDate.toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Button variant="outline" size="sm">
                        Edit
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => handleClosePosition(position.id)}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <div className="text-sm text-muted-foreground">Quantity</div>
                      <div className="font-medium">{position.quantity.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Entry Price</div>
                      <div className="font-medium">{formatCurrency(position.entryPrice)}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Current Price</div>
                      <div className="font-medium">{formatCurrency(position.currentPrice)}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Total Value</div>
                      <div className="font-medium">{formatCurrency(position.totalValue)}</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                    <div>
                      <div className="text-sm text-muted-foreground">Unrealized P&L</div>
                      <div className={`font-bold ${position.unrealizedPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {formatCurrency(position.unrealizedPnL)}
                        <span className="text-sm ml-1">
                          ({formatPercent(position.unrealizedPnLPercent)})
                        </span>
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Realized P&L</div>
                      <div className={`font-medium ${position.realizedPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {formatCurrency(position.realizedPnL)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Stop Loss</div>
                      <div className="font-medium">
                        {position.stopLoss ? formatCurrency(position.stopLoss) : 'Not set'}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Take Profit</div>
                      <div className="font-medium">
                        {position.takeProfit ? formatCurrency(position.takeProfit) : 'Not set'}
                      </div>
                    </div>
                  </div>

                  {/* Risk Indicators */}
                  {position.riskLevel === 'high' && (
                    <div className="mt-4 flex items-center space-x-2 p-2 bg-red-50 dark:bg-red-900/20 rounded-md">
                      <AlertTriangle className="h-4 w-4 text-red-500" />
                      <span className="text-sm text-red-700 dark:text-red-300">
                        High risk position - consider setting stop loss
                      </span>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
