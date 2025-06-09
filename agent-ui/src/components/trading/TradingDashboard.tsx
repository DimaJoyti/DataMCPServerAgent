'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Activity,
  BarChart3,
  PieChart,
  Settings,
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap,
  Target,
  Shield,
  Eye
} from 'lucide-react'
import { MarketDataPanel } from './MarketDataPanel'
import { OrderEntry } from './OrderEntry'
import { OrderBook } from './OrderBook'
import { PositionManager } from './PositionManager'
import { RiskDashboard } from './RiskDashboard'
import { PerformanceAnalytics } from './PerformanceAnalytics'
import { SystemMonitor } from './SystemMonitor'

interface TradingMetrics {
  totalPnL: number
  dailyPnL: number
  totalVolume: number
  ordersExecuted: number
  fillRate: number
  avgLatency: number
  systemStatus: 'online' | 'degraded' | 'offline'
  riskLevel: 'low' | 'medium' | 'high'
}

export const TradingDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview')
  const [metrics, setMetrics] = useState<TradingMetrics>({
    totalPnL: 125420.50,
    dailyPnL: 8750.25,
    totalVolume: 2450000,
    ordersExecuted: 1247,
    fillRate: 98.5,
    avgLatency: 0.8,
    systemStatus: 'online',
    riskLevel: 'medium'
  })

  const [isConnected, setIsConnected] = useState(true)

  useEffect(() => {
    // Simulate real-time updates
    const interval = setInterval(() => {
      setMetrics(prev => ({
        ...prev,
        dailyPnL: prev.dailyPnL + (Math.random() - 0.5) * 100,
        ordersExecuted: prev.ordersExecuted + Math.floor(Math.random() * 3),
        avgLatency: 0.5 + Math.random() * 1.0
      }))
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'text-green-500'
      case 'degraded': return 'text-yellow-500'
      case 'offline': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-500'
      case 'medium': return 'text-yellow-500'
      case 'high': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  return (
    <div className="h-full bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 p-6">
      <div className="max-w-7xl mx-auto h-full flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg">
              <TrendingUp className="h-7 w-7 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-300 bg-clip-text text-transparent">
                Institutional Trading System
              </h1>
              <p className="text-muted-foreground">High-frequency trading & risk management platform</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
              <span className="text-sm font-medium">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <Button variant="outline" size="sm">
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </Button>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4 mb-6">
          <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <DollarSign className="h-5 w-5 text-green-500" />
                <div>
                  <p className="text-xs text-muted-foreground">Total P&L</p>
                  <p className="text-lg font-bold text-green-500">
                    ${metrics.totalPnL.toLocaleString()}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <TrendingUp className={`h-5 w-5 ${metrics.dailyPnL >= 0 ? 'text-green-500' : 'text-red-500'}`} />
                <div>
                  <p className="text-xs text-muted-foreground">Daily P&L</p>
                  <p className={`text-lg font-bold ${metrics.dailyPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    ${metrics.dailyPnL.toFixed(2)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <BarChart3 className="h-5 w-5 text-blue-500" />
                <div>
                  <p className="text-xs text-muted-foreground">Volume</p>
                  <p className="text-lg font-bold text-blue-500">
                    ${(metrics.totalVolume / 1000000).toFixed(1)}M
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Target className="h-5 w-5 text-purple-500" />
                <div>
                  <p className="text-xs text-muted-foreground">Orders</p>
                  <p className="text-lg font-bold text-purple-500">
                    {metrics.ordersExecuted.toLocaleString()}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-5 w-5 text-emerald-500" />
                <div>
                  <p className="text-xs text-muted-foreground">Fill Rate</p>
                  <p className="text-lg font-bold text-emerald-500">
                    {metrics.fillRate}%
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Zap className="h-5 w-5 text-yellow-500" />
                <div>
                  <p className="text-xs text-muted-foreground">Latency</p>
                  <p className="text-lg font-bold text-yellow-500">
                    {metrics.avgLatency.toFixed(1)}ms
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Activity className={`h-5 w-5 ${getStatusColor(metrics.systemStatus)}`} />
                <div>
                  <p className="text-xs text-muted-foreground">System</p>
                  <p className={`text-lg font-bold ${getStatusColor(metrics.systemStatus)} capitalize`}>
                    {metrics.systemStatus}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Shield className={`h-5 w-5 ${getRiskColor(metrics.riskLevel)}`} />
                <div>
                  <p className="text-xs text-muted-foreground">Risk</p>
                  <p className={`text-lg font-bold ${getRiskColor(metrics.riskLevel)} capitalize`}>
                    {metrics.riskLevel}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <div className="flex-1 min-h-0">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
            <TabsList className="grid w-full grid-cols-7 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
              <TabsTrigger value="overview" className="flex items-center space-x-2">
                <Eye className="h-4 w-4" />
                <span>Overview</span>
              </TabsTrigger>
              <TabsTrigger value="market-data" className="flex items-center space-x-2">
                <BarChart3 className="h-4 w-4" />
                <span>Market Data</span>
              </TabsTrigger>
              <TabsTrigger value="orders" className="flex items-center space-x-2">
                <Target className="h-4 w-4" />
                <span>Orders</span>
              </TabsTrigger>
              <TabsTrigger value="positions" className="flex items-center space-x-2">
                <PieChart className="h-4 w-4" />
                <span>Positions</span>
              </TabsTrigger>
              <TabsTrigger value="risk" className="flex items-center space-x-2">
                <Shield className="h-4 w-4" />
                <span>Risk</span>
              </TabsTrigger>
              <TabsTrigger value="analytics" className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4" />
                <span>Analytics</span>
              </TabsTrigger>
              <TabsTrigger value="system" className="flex items-center space-x-2">
                <Activity className="h-4 w-4" />
                <span>System</span>
              </TabsTrigger>
            </TabsList>

            <div className="flex-1 mt-4">
              <TabsContent value="overview" className="h-full">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
                  <MarketDataPanel />
                  <OrderEntry />
                </div>
              </TabsContent>

              <TabsContent value="market-data" className="h-full">
                <MarketDataPanel />
              </TabsContent>

              <TabsContent value="orders" className="h-full">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
                  <OrderEntry />
                  <OrderBook />
                </div>
              </TabsContent>

              <TabsContent value="positions" className="h-full">
                <PositionManager />
              </TabsContent>

              <TabsContent value="risk" className="h-full">
                <RiskDashboard />
              </TabsContent>

              <TabsContent value="analytics" className="h-full">
                <PerformanceAnalytics />
              </TabsContent>

              <TabsContent value="system" className="h-full">
                <SystemMonitor />
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </div>
    </div>
  )
}
