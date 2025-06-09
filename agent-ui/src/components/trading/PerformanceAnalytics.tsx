'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  TrendingUp,
  BarChart3,
  PieChart,
  Target,
  Clock,
  DollarSign,
  Activity,
  Award,
  Download
} from 'lucide-react'

interface PerformanceMetric {
  name: string
  value: string
  change: number
  changePercent: number
  benchmark?: string
}

interface TradeAnalysis {
  totalTrades: number
  winningTrades: number
  losingTrades: number
  winRate: number
  avgWin: number
  avgLoss: number
  profitFactor: number
  largestWin: number
  largestLoss: number
}

interface ExecutionMetric {
  metric: string
  value: string
  target: string
  status: 'good' | 'warning' | 'poor'
}

export const PerformanceAnalytics: React.FC = () => {
  const [timeframe, setTimeframe] = useState('1M')
  const [activeTab, setActiveTab] = useState('overview')

  const performanceMetrics: PerformanceMetric[] = [
    {
      name: 'Total Return',
      value: '+12.45%',
      change: 2.1,
      changePercent: 0.25,
      benchmark: 'S&P 500: +8.2%'
    },
    {
      name: 'Sharpe Ratio',
      value: '1.85',
      change: 0.15,
      changePercent: 8.8
    },
    {
      name: 'Max Drawdown',
      value: '-5.2%',
      change: -0.8,
      changePercent: -13.3
    },
    {
      name: 'Volatility',
      value: '18.5%',
      change: 1.2,
      changePercent: 6.9
    },
    {
      name: 'Alpha',
      value: '4.25%',
      change: 0.5,
      changePercent: 13.3
    },
    {
      name: 'Beta',
      value: '1.15',
      change: -0.05,
      changePercent: -4.2
    }
  ]

  const tradeAnalysis: TradeAnalysis = {
    totalTrades: 247,
    winningTrades: 152,
    losingTrades: 95,
    winRate: 61.5,
    avgWin: 1250.75,
    avgLoss: -875.25,
    profitFactor: 1.85,
    largestWin: 8750.00,
    largestLoss: -3250.00
  }

  const executionMetrics: ExecutionMetric[] = [
    {
      metric: 'Average Fill Time',
      value: '0.8ms',
      target: '<1ms',
      status: 'good'
    },
    {
      metric: 'Fill Rate',
      value: '98.5%',
      target: '>95%',
      status: 'good'
    },
    {
      metric: 'Slippage',
      value: '0.02%',
      target: '<0.05%',
      status: 'good'
    },
    {
      metric: 'Market Impact',
      value: '0.15%',
      target: '<0.20%',
      status: 'good'
    },
    {
      metric: 'Implementation Shortfall',
      value: '0.18%',
      target: '<0.25%',
      status: 'good'
    },
    {
      metric: 'Order Rejection Rate',
      value: '0.3%',
      target: '<1%',
      status: 'good'
    }
  ]

  const timeframes = [
    { value: '1D', label: '1 Day' },
    { value: '1W', label: '1 Week' },
    { value: '1M', label: '1 Month' },
    { value: '3M', label: '3 Months' },
    { value: '6M', label: '6 Months' },
    { value: '1Y', label: '1 Year' },
    { value: 'YTD', label: 'Year to Date' },
    { value: 'ALL', label: 'All Time' }
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'text-green-500'
      case 'warning': return 'text-yellow-500'
      case 'poor': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'good': return 'default'
      case 'warning': return 'secondary'
      case 'poor': return 'destructive'
      default: return 'outline'
    }
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount)
  }

  return (
    <div className="h-full space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Performance Analytics</h2>
          <p className="text-muted-foreground">Comprehensive trading performance analysis</p>
        </div>
        <div className="flex items-center space-x-4">
          <Select value={timeframe} onValueChange={setTimeframe}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {timeframes.map(tf => (
                <SelectItem key={tf.value} value={tf.value}>
                  {tf.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="trades">Trade Analysis</TabsTrigger>
          <TabsTrigger value="execution">Execution Quality</TabsTrigger>
          <TabsTrigger value="attribution">Attribution</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Performance Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {performanceMetrics.map((metric, index) => (
              <Card key={index} className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-muted-foreground">{metric.name}</span>
                    <TrendingUp className={`h-4 w-4 ${metric.change >= 0 ? 'text-green-500' : 'text-red-500'}`} />
                  </div>
                  <div className="space-y-1">
                    <div className="text-2xl font-bold">{metric.value}</div>
                    <div className={`text-sm ${metric.change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {metric.change >= 0 ? '+' : ''}{metric.change} ({metric.changePercent >= 0 ? '+' : ''}{metric.changePercent.toFixed(1)}%)
                    </div>
                    {metric.benchmark && (
                      <div className="text-xs text-muted-foreground">{metric.benchmark}</div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Performance Chart Placeholder */}
          <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
            <CardHeader>
              <CardTitle>Portfolio Performance</CardTitle>
              <CardDescription>Cumulative returns over time</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Performance chart visualization</p>
                  <p className="text-sm">Chart component integration coming soon</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trades" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Trade Statistics */}
            <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Target className="h-5 w-5" />
                  <span>Trade Statistics</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-muted-foreground">Total Trades</div>
                    <div className="text-2xl font-bold">{tradeAnalysis.totalTrades}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Win Rate</div>
                    <div className="text-2xl font-bold text-green-500">{tradeAnalysis.winRate}%</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Winning Trades</div>
                    <div className="text-lg font-medium text-green-500">{tradeAnalysis.winningTrades}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Losing Trades</div>
                    <div className="text-lg font-medium text-red-500">{tradeAnalysis.losingTrades}</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Profit/Loss Analysis */}
            <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <DollarSign className="h-5 w-5" />
                  <span>P&L Analysis</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-muted-foreground">Avg Win</div>
                    <div className="text-lg font-bold text-green-500">
                      {formatCurrency(tradeAnalysis.avgWin)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Avg Loss</div>
                    <div className="text-lg font-bold text-red-500">
                      {formatCurrency(tradeAnalysis.avgLoss)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Largest Win</div>
                    <div className="text-lg font-medium text-green-500">
                      {formatCurrency(tradeAnalysis.largestWin)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Largest Loss</div>
                    <div className="text-lg font-medium text-red-500">
                      {formatCurrency(tradeAnalysis.largestLoss)}
                    </div>
                  </div>
                </div>
                <div className="pt-4 border-t">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Profit Factor</span>
                    <span className="text-lg font-bold">{tradeAnalysis.profitFactor}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="execution" className="space-y-6">
          <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Clock className="h-5 w-5" />
                <span>Execution Quality Metrics</span>
              </CardTitle>
              <CardDescription>Order execution performance and efficiency</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {executionMetrics.map((metric, index) => (
                  <div key={index} className="border border-border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{metric.metric}</span>
                      <Badge variant={getStatusBadgeVariant(metric.status)}>
                        {metric.status.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="space-y-1">
                      <div className={`text-xl font-bold ${getStatusColor(metric.status)}`}>
                        {metric.value}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Target: {metric.target}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="attribution" className="space-y-6">
          <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <PieChart className="h-5 w-5" />
                <span>Performance Attribution</span>
              </CardTitle>
              <CardDescription>Breakdown of returns by strategy and asset class</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <Award className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Performance attribution analysis</p>
                  <p className="text-sm">Advanced analytics coming soon</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
