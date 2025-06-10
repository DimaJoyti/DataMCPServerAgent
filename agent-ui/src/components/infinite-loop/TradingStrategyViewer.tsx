'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Target, 
  Shield, 
  Zap,
  Play,
  Pause,
  Square,
  Eye,
  Download,
  Settings,
  AlertTriangle,
  CheckCircle
} from 'lucide-react'
import { useInfiniteLoopStore } from '@/store'

interface TradingStrategy {
  id: string
  name: string
  type: 'momentum' | 'mean_reversion' | 'arbitrage' | 'ml_based'
  status: 'generated' | 'backtesting' | 'deployed' | 'paused' | 'stopped'
  performance: {
    sharpe_ratio: number
    total_return: number
    max_drawdown: number
    win_rate: number
    profit_factor: number
    calmar_ratio: number
  }
  risk_metrics: {
    var_95: number
    expected_shortfall: number
    beta: number
    alpha: number
  }
  deployment: {
    allocation: number
    max_position_size: number
    stop_loss: number
    take_profit: number
  }
  backtest_results: {
    start_date: string
    end_date: string
    total_trades: number
    winning_trades: number
    losing_trades: number
    avg_trade_duration: number
    largest_win: number
    largest_loss: number
  }
  created_at: string
  last_updated: string
}

export const TradingStrategyViewer: React.FC = () => {
  const { currentSession } = useInfiniteLoopStore()
  const [strategies, setStrategies] = useState<TradingStrategy[]>([])
  const [selectedStrategy, setSelectedStrategy] = useState<TradingStrategy | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    // Load strategies from API or session
    loadStrategies()
  }, [currentSession])

  const loadStrategies = async () => {
    setLoading(true)
    try {
      // Mock data for demonstration
      const mockStrategies: TradingStrategy[] = [
        {
          id: 'strategy_1',
          name: 'Momentum Breakout v2.1',
          type: 'momentum',
          status: 'deployed',
          performance: {
            sharpe_ratio: 2.34,
            total_return: 0.187,
            max_drawdown: -0.045,
            win_rate: 0.68,
            profit_factor: 1.85,
            calmar_ratio: 5.2
          },
          risk_metrics: {
            var_95: -0.023,
            expected_shortfall: -0.031,
            beta: 0.85,
            alpha: 0.12
          },
          deployment: {
            allocation: 0.15,
            max_position_size: 0.05,
            stop_loss: 0.02,
            take_profit: 0.06
          },
          backtest_results: {
            start_date: '2024-01-01',
            end_date: '2024-01-31',
            total_trades: 156,
            winning_trades: 106,
            losing_trades: 50,
            avg_trade_duration: 4.2,
            largest_win: 0.087,
            largest_loss: -0.023
          },
          created_at: '2024-01-15T10:30:00Z',
          last_updated: '2024-01-20T14:45:00Z'
        },
        {
          id: 'strategy_2',
          name: 'Mean Reversion RSI',
          type: 'mean_reversion',
          status: 'backtesting',
          performance: {
            sharpe_ratio: 1.89,
            total_return: 0.142,
            max_drawdown: -0.067,
            win_rate: 0.72,
            profit_factor: 1.64,
            calmar_ratio: 2.1
          },
          risk_metrics: {
            var_95: -0.028,
            expected_shortfall: -0.039,
            beta: 0.92,
            alpha: 0.08
          },
          deployment: {
            allocation: 0.10,
            max_position_size: 0.03,
            stop_loss: 0.015,
            take_profit: 0.045
          },
          backtest_results: {
            start_date: '2024-01-01',
            end_date: '2024-01-31',
            total_trades: 203,
            winning_trades: 146,
            losing_trades: 57,
            avg_trade_duration: 6.8,
            largest_win: 0.054,
            largest_loss: -0.019
          },
          created_at: '2024-01-18T09:15:00Z',
          last_updated: '2024-01-20T16:20:00Z'
        }
      ]
      
      setStrategies(mockStrategies)
      if (mockStrategies.length > 0) {
        setSelectedStrategy(mockStrategies[0])
      }
    } catch (error) {
      console.error('Error loading strategies:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': return 'bg-green-500'
      case 'backtesting': return 'bg-blue-500'
      case 'paused': return 'bg-yellow-500'
      case 'stopped': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'deployed': return <Play className="h-4 w-4" />
      case 'backtesting': return <Zap className="h-4 w-4" />
      case 'paused': return <Pause className="h-4 w-4" />
      case 'stopped': return <Square className="h-4 w-4" />
      default: return <Eye className="h-4 w-4" />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'momentum': return 'bg-blue-100 text-blue-800'
      case 'mean_reversion': return 'bg-green-100 text-green-800'
      case 'arbitrage': return 'bg-purple-100 text-purple-800'
      case 'ml_based': return 'bg-orange-100 text-orange-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-muted-foreground">Loading trading strategies...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex">
      {/* Strategy List */}
      <div className="w-1/3 border-r border-border p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Generated Strategies</h3>
          <Badge variant="outline">{strategies.length}</Badge>
        </div>
        
        <div className="space-y-2">
          {strategies.map((strategy) => (
            <Card 
              key={strategy.id}
              className={`cursor-pointer transition-all ${
                selectedStrategy?.id === strategy.id ? 'ring-2 ring-blue-500' : ''
              }`}
              onClick={() => setSelectedStrategy(strategy)}
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium truncate">{strategy.name}</h4>
                  <div className="flex items-center space-x-1">
                    <div className={`w-2 h-2 rounded-full ${getStatusColor(strategy.status)}`} />
                    {getStatusIcon(strategy.status)}
                  </div>
                </div>
                
                <div className="flex items-center justify-between mb-2">
                  <Badge className={getTypeColor(strategy.type)}>
                    {strategy.type.replace('_', ' ')}
                  </Badge>
                  <span className="text-sm text-muted-foreground">
                    {(strategy.performance.total_return * 100).toFixed(1)}%
                  </span>
                </div>
                
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-muted-foreground">Sharpe:</span>
                    <span className="ml-1 font-medium">{strategy.performance.sharpe_ratio.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Win Rate:</span>
                    <span className="ml-1 font-medium">{(strategy.performance.win_rate * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Strategy Details */}
      <div className="flex-1 p-6">
        {selectedStrategy ? (
          <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold">{selectedStrategy.name}</h2>
                <div className="flex items-center space-x-2 mt-1">
                  <Badge className={getTypeColor(selectedStrategy.type)}>
                    {selectedStrategy.type.replace('_', ' ')}
                  </Badge>
                  <Badge variant="outline" className="flex items-center space-x-1">
                    <div className={`w-2 h-2 rounded-full ${getStatusColor(selectedStrategy.status)}`} />
                    <span className="capitalize">{selectedStrategy.status}</span>
                  </Badge>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <Button variant="outline" size="sm">
                  <Download className="mr-2 h-4 w-4" />
                  Export
                </Button>
                <Button variant="outline" size="sm">
                  <Settings className="mr-2 h-4 w-4" />
                  Configure
                </Button>
                {selectedStrategy.status === 'deployed' ? (
                  <Button variant="outline" size="sm">
                    <Pause className="mr-2 h-4 w-4" />
                    Pause
                  </Button>
                ) : (
                  <Button size="sm">
                    <Play className="mr-2 h-4 w-4" />
                    Deploy
                  </Button>
                )}
              </div>
            </div>

            <Tabs defaultValue="performance" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="performance">Performance</TabsTrigger>
                <TabsTrigger value="risk">Risk Analysis</TabsTrigger>
                <TabsTrigger value="backtest">Backtest</TabsTrigger>
                <TabsTrigger value="deployment">Deployment</TabsTrigger>
              </TabsList>

              <TabsContent value="performance" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium flex items-center space-x-2">
                        <TrendingUp className="h-4 w-4 text-green-500" />
                        <span>Total Return</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-green-600">
                        {(selectedStrategy.performance.total_return * 100).toFixed(2)}%
                      </div>
                      <p className="text-sm text-muted-foreground">Since inception</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium flex items-center space-x-2">
                        <Target className="h-4 w-4 text-blue-500" />
                        <span>Sharpe Ratio</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">
                        {selectedStrategy.performance.sharpe_ratio.toFixed(2)}
                      </div>
                      <p className="text-sm text-muted-foreground">Risk-adjusted return</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium flex items-center space-x-2">
                        <TrendingDown className="h-4 w-4 text-red-500" />
                        <span>Max Drawdown</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-red-600">
                        {(selectedStrategy.performance.max_drawdown * 100).toFixed(2)}%
                      </div>
                      <p className="text-sm text-muted-foreground">Maximum loss</p>
                    </CardContent>
                  </Card>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Success Rate</span>
                          <span>{(selectedStrategy.performance.win_rate * 100).toFixed(1)}%</span>
                        </div>
                        <Progress value={selectedStrategy.performance.win_rate * 100} />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium">Profit Factor</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">
                        {selectedStrategy.performance.profit_factor.toFixed(2)}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Gross profit / Gross loss
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="risk" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium flex items-center space-x-2">
                        <Shield className="h-4 w-4 text-orange-500" />
                        <span>Value at Risk (95%)</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-orange-600">
                        {(selectedStrategy.risk_metrics.var_95 * 100).toFixed(2)}%
                      </div>
                      <p className="text-sm text-muted-foreground">Daily VaR</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium">Expected Shortfall</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-red-600">
                        {(selectedStrategy.risk_metrics.expected_shortfall * 100).toFixed(2)}%
                      </div>
                      <p className="text-sm text-muted-foreground">Conditional VaR</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium">Beta</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">
                        {selectedStrategy.risk_metrics.beta.toFixed(2)}
                      </div>
                      <p className="text-sm text-muted-foreground">Market sensitivity</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium">Alpha</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-green-600">
                        {(selectedStrategy.risk_metrics.alpha * 100).toFixed(2)}%
                      </div>
                      <p className="text-sm text-muted-foreground">Excess return</p>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="backtest" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium">Total Trades</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">
                        {selectedStrategy.backtest_results.total_trades}
                      </div>
                      <div className="flex items-center space-x-2 text-sm">
                        <CheckCircle className="h-3 w-3 text-green-500" />
                        <span>{selectedStrategy.backtest_results.winning_trades} wins</span>
                        <AlertTriangle className="h-3 w-3 text-red-500" />
                        <span>{selectedStrategy.backtest_results.losing_trades} losses</span>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium">Avg Trade Duration</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">
                        {selectedStrategy.backtest_results.avg_trade_duration.toFixed(1)}h
                      </div>
                      <p className="text-sm text-muted-foreground">Average holding period</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium">Best/Worst Trade</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-1">
                        <div className="flex justify-between">
                          <span className="text-sm text-green-600">Best:</span>
                          <span className="text-sm font-medium text-green-600">
                            +{(selectedStrategy.backtest_results.largest_win * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-red-600">Worst:</span>
                          <span className="text-sm font-medium text-red-600">
                            {(selectedStrategy.backtest_results.largest_loss * 100).toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="deployment" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium">Portfolio Allocation</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Allocation</span>
                          <span>{(selectedStrategy.deployment.allocation * 100).toFixed(1)}%</span>
                        </div>
                        <Progress value={selectedStrategy.deployment.allocation * 100} />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium">Position Sizing</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Max Position</span>
                          <span>{(selectedStrategy.deployment.max_position_size * 100).toFixed(1)}%</span>
                        </div>
                        <Progress value={selectedStrategy.deployment.max_position_size * 100} />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium">Risk Controls</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Stop Loss:</span>
                          <span className="text-red-600">
                            -{(selectedStrategy.deployment.stop_loss * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Take Profit:</span>
                          <span className="text-green-600">
                            +{(selectedStrategy.deployment.take_profit * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-4">
              <DollarSign className="h-12 w-12 mx-auto text-muted-foreground" />
              <div>
                <h3 className="text-lg font-semibold">No Strategy Selected</h3>
                <p className="text-muted-foreground">Select a strategy from the list to view details</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
