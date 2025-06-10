'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  ScatterPlot,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { 
  TrendingUp, 
  BarChart3, 
  PieChart as PieChartIcon, 
  Activity, 
  Target,
  Download,
  RefreshCw
} from 'lucide-react'
import { useInfiniteLoopStore } from '@/store'

interface ChartData {
  timestamp: string
  value: number
  label?: string
  category?: string
}

interface PerformanceData {
  equity_curve: ChartData[]
  drawdown_curve: ChartData[]
  returns_distribution: ChartData[]
  strategy_comparison: ChartData[]
  risk_metrics: ChartData[]
  trade_analysis: ChartData[]
}

export const PerformanceCharts: React.FC = () => {
  const { currentSession } = useInfiniteLoopStore()
  const [timeframe, setTimeframe] = useState<string>('1M')
  const [chartType, setChartType] = useState<string>('equity')
  const [performanceData, setPerformanceData] = useState<PerformanceData | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadPerformanceData()
  }, [currentSession, timeframe])

  const loadPerformanceData = async () => {
    setLoading(true)
    try {
      // Mock performance data
      const mockData: PerformanceData = {
        equity_curve: generateEquityCurve(),
        drawdown_curve: generateDrawdownCurve(),
        returns_distribution: generateReturnsDistribution(),
        strategy_comparison: generateStrategyComparison(),
        risk_metrics: generateRiskMetrics(),
        trade_analysis: generateTradeAnalysis()
      }
      
      setPerformanceData(mockData)
    } catch (error) {
      console.error('Error loading performance data:', error)
    } finally {
      setLoading(false)
    }
  }

  const generateEquityCurve = (): ChartData[] => {
    const data: ChartData[] = []
    let value = 100000 // Starting capital
    const days = timeframe === '1W' ? 7 : timeframe === '1M' ? 30 : timeframe === '3M' ? 90 : 365
    
    for (let i = 0; i < days; i++) {
      const date = new Date()
      date.setDate(date.getDate() - (days - i))
      
      // Simulate realistic equity curve with some volatility
      const dailyReturn = (Math.random() - 0.45) * 0.02 // Slight positive bias
      value *= (1 + dailyReturn)
      
      data.push({
        timestamp: date.toISOString().split('T')[0],
        value: Math.round(value),
        label: `$${Math.round(value).toLocaleString()}`
      })
    }
    
    return data
  }

  const generateDrawdownCurve = (): ChartData[] => {
    const data: ChartData[] = []
    let peak = 100000
    let current = 100000
    const days = timeframe === '1W' ? 7 : timeframe === '1M' ? 30 : timeframe === '3M' ? 90 : 365
    
    for (let i = 0; i < days; i++) {
      const date = new Date()
      date.setDate(date.getDate() - (days - i))
      
      const dailyReturn = (Math.random() - 0.45) * 0.02
      current *= (1 + dailyReturn)
      
      if (current > peak) {
        peak = current
      }
      
      const drawdown = ((current - peak) / peak) * 100
      
      data.push({
        timestamp: date.toISOString().split('T')[0],
        value: drawdown,
        label: `${drawdown.toFixed(2)}%`
      })
    }
    
    return data
  }

  const generateReturnsDistribution = (): ChartData[] => {
    const data: ChartData[] = []
    const buckets = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    
    buckets.forEach(bucket => {
      // Normal distribution centered around 0.1% daily return
      const frequency = Math.exp(-Math.pow(bucket - 0.1, 2) / 2) * 100 + Math.random() * 20
      
      data.push({
        timestamp: `${bucket}%`,
        value: Math.round(frequency),
        label: `${bucket}% to ${bucket + 1}%`,
        category: bucket < 0 ? 'negative' : 'positive'
      })
    })
    
    return data
  }

  const generateStrategyComparison = (): ChartData[] => {
    const strategies = ['Momentum', 'Mean Reversion', 'Arbitrage', 'ML-Based', 'Benchmark']
    const metrics = ['Return', 'Sharpe', 'Max DD', 'Win Rate']
    const data: ChartData[] = []
    
    strategies.forEach(strategy => {
      metrics.forEach(metric => {
        let value = 0
        switch (metric) {
          case 'Return':
            value = Math.random() * 30 + 5 // 5-35%
            break
          case 'Sharpe':
            value = Math.random() * 2 + 0.5 // 0.5-2.5
            break
          case 'Max DD':
            value = -(Math.random() * 15 + 2) // -2% to -17%
            break
          case 'Win Rate':
            value = Math.random() * 40 + 50 // 50-90%
            break
        }
        
        data.push({
          timestamp: strategy,
          value: Math.round(value * 100) / 100,
          label: metric,
          category: metric
        })
      })
    })
    
    return data
  }

  const generateRiskMetrics = (): ChartData[] => {
    const metrics = ['VaR 95%', 'VaR 99%', 'Expected Shortfall', 'Beta', 'Alpha']
    
    return metrics.map(metric => {
      let value = 0
      switch (metric) {
        case 'VaR 95%':
          value = -(Math.random() * 3 + 1) // -1% to -4%
          break
        case 'VaR 99%':
          value = -(Math.random() * 5 + 2) // -2% to -7%
          break
        case 'Expected Shortfall':
          value = -(Math.random() * 6 + 3) // -3% to -9%
          break
        case 'Beta':
          value = Math.random() * 1.5 + 0.3 // 0.3 to 1.8
          break
        case 'Alpha':
          value = Math.random() * 0.2 - 0.05 // -5% to 15%
          break
      }
      
      return {
        timestamp: metric,
        value: Math.round(value * 100) / 100,
        label: `${value.toFixed(2)}${metric.includes('%') ? '%' : ''}`
      }
    })
  }

  const generateTradeAnalysis = (): ChartData[] => {
    const hours = Array.from({ length: 24 }, (_, i) => i)
    
    return hours.map(hour => {
      // Simulate trading activity with higher activity during market hours
      const isMarketHours = hour >= 9 && hour <= 16
      const baseActivity = isMarketHours ? 50 : 10
      const activity = baseActivity + Math.random() * 30
      
      return {
        timestamp: `${hour}:00`,
        value: Math.round(activity),
        label: `${Math.round(activity)} trades`
      }
    })
  }

  const renderChart = () => {
    if (!performanceData) return null

    switch (chartType) {
      case 'equity':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={performanceData.equity_curve}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip formatter={(value) => [`$${Number(value).toLocaleString()}`, 'Portfolio Value']} />
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke="#3b82f6" 
                fill="#3b82f6" 
                fillOpacity={0.1}
              />
            </AreaChart>
          </ResponsiveContainer>
        )

      case 'drawdown':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={performanceData.drawdown_curve}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip formatter={(value) => [`${Number(value).toFixed(2)}%`, 'Drawdown']} />
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke="#ef4444" 
                fill="#ef4444" 
                fillOpacity={0.1}
              />
            </AreaChart>
          </ResponsiveContainer>
        )

      case 'returns':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={performanceData.returns_distribution}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Bar 
                dataKey="value" 
                fill={(entry: any) => entry.category === 'negative' ? '#ef4444' : '#10b981'}
              />
            </BarChart>
          </ResponsiveContainer>
        )

      case 'comparison':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={performanceData.strategy_comparison}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="value" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        )

      case 'risk':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={performanceData.risk_metrics} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="timestamp" type="category" width={120} />
              <Tooltip />
              <Bar dataKey="value" fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        )

      case 'trades':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={performanceData.trade_analysis}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#8b5cf6" 
                strokeWidth={2}
                dot={{ fill: '#8b5cf6', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )

      default:
        return null
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center space-y-4">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-muted-foreground">Loading performance charts...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5" />
            <h3 className="text-lg font-semibold">Performance Charts</h3>
          </div>
          <Badge variant="outline">Real-time</Badge>
        </div>
        
        <div className="flex items-center space-x-2">
          <Select value={timeframe} onValueChange={setTimeframe}>
            <SelectTrigger className="w-24">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1W">1W</SelectItem>
              <SelectItem value="1M">1M</SelectItem>
              <SelectItem value="3M">3M</SelectItem>
              <SelectItem value="1Y">1Y</SelectItem>
            </SelectContent>
          </Select>
          
          <Button variant="outline" size="sm" onClick={loadPerformanceData}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
          
          <Button variant="outline" size="sm">
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Chart Type Selector */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
        {[
          { key: 'equity', label: 'Equity Curve', icon: TrendingUp },
          { key: 'drawdown', label: 'Drawdown', icon: Activity },
          { key: 'returns', label: 'Returns Dist.', icon: BarChart3 },
          { key: 'comparison', label: 'Strategy Comp.', icon: Target },
          { key: 'risk', label: 'Risk Metrics', icon: PieChartIcon },
          { key: 'trades', label: 'Trade Analysis', icon: Activity }
        ].map(({ key, label, icon: Icon }) => (
          <Button
            key={key}
            variant={chartType === key ? 'default' : 'outline'}
            size="sm"
            onClick={() => setChartType(key)}
            className="flex flex-col items-center space-y-1 h-auto py-3"
          >
            <Icon className="h-4 w-4" />
            <span className="text-xs">{label}</span>
          </Button>
        ))}
      </div>

      {/* Main Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5" />
            <span>
              {chartType === 'equity' && 'Portfolio Equity Curve'}
              {chartType === 'drawdown' && 'Drawdown Analysis'}
              {chartType === 'returns' && 'Returns Distribution'}
              {chartType === 'comparison' && 'Strategy Comparison'}
              {chartType === 'risk' && 'Risk Metrics'}
              {chartType === 'trades' && 'Trading Activity by Hour'}
            </span>
          </CardTitle>
          <CardDescription>
            {chartType === 'equity' && 'Portfolio value over time showing cumulative performance'}
            {chartType === 'drawdown' && 'Maximum drawdown from peak portfolio value'}
            {chartType === 'returns' && 'Distribution of daily returns showing risk profile'}
            {chartType === 'comparison' && 'Performance comparison across different strategies'}
            {chartType === 'risk' && 'Risk metrics including VaR, beta, and alpha'}
            {chartType === 'trades' && 'Trading activity distribution throughout the day'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {renderChart()}
        </CardContent>
      </Card>

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Return</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">+18.7%</div>
            <p className="text-sm text-muted-foreground">Since inception</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2.34</div>
            <p className="text-sm text-muted-foreground">Risk-adjusted return</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Max Drawdown</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">-4.5%</div>
            <p className="text-sm text-muted-foreground">Maximum loss</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">68%</div>
            <p className="text-sm text-muted-foreground">Successful trades</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
