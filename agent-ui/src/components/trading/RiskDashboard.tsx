'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import {
  Shield,
  AlertTriangle,
  TrendingDown,
  Target,
  Activity,
  BarChart3,
  Settings,
  CheckCircle,
  XCircle
} from 'lucide-react'

interface RiskMetric {
  name: string
  value: number
  limit: number
  status: 'safe' | 'warning' | 'danger'
  unit: string
}

interface RiskAlert {
  id: string
  type: 'limit_breach' | 'concentration' | 'volatility' | 'correlation'
  severity: 'low' | 'medium' | 'high'
  message: string
  timestamp: Date
  acknowledged: boolean
}

export const RiskDashboard: React.FC = () => {
  const [riskMetrics, setRiskMetrics] = useState<RiskMetric[]>([
    {
      name: 'Value at Risk (1D)',
      value: 15420.50,
      limit: 25000.00,
      status: 'safe',
      unit: '$'
    },
    {
      name: 'Maximum Drawdown',
      value: 8.5,
      limit: 15.0,
      status: 'safe',
      unit: '%'
    },
    {
      name: 'Portfolio Beta',
      value: 1.25,
      limit: 1.50,
      status: 'warning',
      unit: ''
    },
    {
      name: 'Concentration Risk',
      value: 35.2,
      limit: 40.0,
      status: 'warning',
      unit: '%'
    },
    {
      name: 'Leverage Ratio',
      value: 2.1,
      limit: 3.0,
      status: 'safe',
      unit: 'x'
    },
    {
      name: 'Sharpe Ratio',
      value: 1.85,
      limit: 1.00,
      status: 'safe',
      unit: ''
    }
  ])

  const [riskAlerts, setRiskAlerts] = useState<RiskAlert[]>([
    {
      id: 'ALERT-001',
      type: 'concentration',
      severity: 'medium',
      message: 'BTC position exceeds 30% of portfolio value',
      timestamp: new Date(Date.now() - 300000),
      acknowledged: false
    },
    {
      id: 'ALERT-002',
      type: 'volatility',
      severity: 'low',
      message: 'Increased volatility detected in tech sector positions',
      timestamp: new Date(Date.now() - 600000),
      acknowledged: false
    },
    {
      id: 'ALERT-003',
      type: 'correlation',
      severity: 'high',
      message: 'High correlation risk between crypto positions',
      timestamp: new Date(Date.now() - 900000),
      acknowledged: true
    }
  ])

  const [overallRiskScore, setOverallRiskScore] = useState(72)

  useEffect(() => {
    // Simulate real-time risk metric updates
    const interval = setInterval(() => {
      setRiskMetrics(prev => prev.map(metric => {
        const change = (Math.random() - 0.5) * metric.value * 0.05
        const newValue = Math.max(0, metric.value + change)
        const percentage = (newValue / metric.limit) * 100
        
        let status: 'safe' | 'warning' | 'danger' = 'safe'
        if (percentage > 90) status = 'danger'
        else if (percentage > 70) status = 'warning'

        return {
          ...metric,
          value: newValue,
          status
        }
      }))

      // Update overall risk score
      setOverallRiskScore(prev => {
        const change = (Math.random() - 0.5) * 5
        return Math.max(0, Math.min(100, prev + change))
      })
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const handleAcknowledgeAlert = (alertId: string) => {
    setRiskAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ))
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'safe': return 'text-green-500'
      case 'warning': return 'text-yellow-500'
      case 'danger': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'safe': return 'default'
      case 'warning': return 'secondary'
      case 'danger': return 'destructive'
      default: return 'outline'
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'text-blue-500'
      case 'medium': return 'text-yellow-500'
      case 'high': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'low': return <Activity className="h-4 w-4 text-blue-500" />
      case 'medium': return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      case 'high': return <XCircle className="h-4 w-4 text-red-500" />
      default: return <Activity className="h-4 w-4 text-gray-500" />
    }
  }

  const getRiskScoreColor = (score: number) => {
    if (score >= 80) return 'text-red-500'
    if (score >= 60) return 'text-yellow-500'
    return 'text-green-500'
  }

  const formatValue = (value: number, unit: string) => {
    if (unit === '$') {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
      }).format(value)
    }
    if (unit === '%') {
      return `${value.toFixed(1)}%`
    }
    if (unit === 'x') {
      return `${value.toFixed(1)}x`
    }
    return value.toFixed(2)
  }

  return (
    <div className="h-full space-y-6 p-6">
      {/* Risk Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg font-bold flex items-center space-x-2">
              <Shield className="h-5 w-5" />
              <span>Overall Risk Score</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className={`text-4xl font-bold ${getRiskScoreColor(overallRiskScore)}`}>
                {overallRiskScore}
              </div>
              <div className="text-sm text-muted-foreground mb-4">out of 100</div>
              <Progress value={overallRiskScore} className="h-3" />
              <div className="text-xs text-muted-foreground mt-2">
                {overallRiskScore >= 80 ? 'High Risk' : 
                 overallRiskScore >= 60 ? 'Medium Risk' : 'Low Risk'}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg font-bold flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5" />
              <span>Active Alerts</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className="text-4xl font-bold text-red-500">
                {riskAlerts.filter(alert => !alert.acknowledged).length}
              </div>
              <div className="text-sm text-muted-foreground mb-4">unacknowledged</div>
              <div className="space-y-1">
                <div className="text-xs">
                  <span className="text-red-500">High: </span>
                  {riskAlerts.filter(a => a.severity === 'high' && !a.acknowledged).length}
                </div>
                <div className="text-xs">
                  <span className="text-yellow-500">Medium: </span>
                  {riskAlerts.filter(a => a.severity === 'medium' && !a.acknowledged).length}
                </div>
                <div className="text-xs">
                  <span className="text-blue-500">Low: </span>
                  {riskAlerts.filter(a => a.severity === 'low' && !a.acknowledged).length}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg font-bold flex items-center space-x-2">
              <Target className="h-5 w-5" />
              <span>Risk Limits</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className="text-4xl font-bold text-green-500">
                {riskMetrics.filter(m => m.status === 'safe').length}
              </div>
              <div className="text-sm text-muted-foreground mb-4">within limits</div>
              <div className="space-y-1">
                <div className="text-xs">
                  <span className="text-green-500">Safe: </span>
                  {riskMetrics.filter(m => m.status === 'safe').length}
                </div>
                <div className="text-xs">
                  <span className="text-yellow-500">Warning: </span>
                  {riskMetrics.filter(m => m.status === 'warning').length}
                </div>
                <div className="text-xs">
                  <span className="text-red-500">Danger: </span>
                  {riskMetrics.filter(m => m.status === 'danger').length}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Risk Metrics */}
      <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-xl font-bold">Risk Metrics</CardTitle>
              <CardDescription>Real-time risk monitoring and limits</CardDescription>
            </div>
            <Button variant="outline" size="sm">
              <Settings className="h-4 w-4 mr-2" />
              Configure Limits
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {riskMetrics.map((metric, index) => (
              <div key={index} className="border border-border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{metric.name}</span>
                  <Badge variant={getStatusBadgeVariant(metric.status)}>
                    {metric.status.toUpperCase()}
                  </Badge>
                </div>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-lg font-bold ${getStatusColor(metric.status)}`}>
                    {formatValue(metric.value, metric.unit)}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    Limit: {formatValue(metric.limit, metric.unit)}
                  </span>
                </div>
                <Progress 
                  value={(metric.value / metric.limit) * 100} 
                  className="h-2"
                />
                <div className="text-xs text-muted-foreground mt-1">
                  {((metric.value / metric.limit) * 100).toFixed(1)}% of limit
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Risk Alerts */}
      <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
        <CardHeader>
          <CardTitle className="text-xl font-bold">Risk Alerts</CardTitle>
          <CardDescription>Recent risk notifications and warnings</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {riskAlerts.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500 opacity-50" />
                <p>No active risk alerts</p>
                <p className="text-sm">All risk metrics are within acceptable limits</p>
              </div>
            ) : (
              riskAlerts.map((alert) => (
                <div 
                  key={alert.id} 
                  className={`border rounded-lg p-4 ${
                    alert.acknowledged ? 'opacity-60 bg-muted/20' : 'bg-background'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {getSeverityIcon(alert.severity)}
                      <div>
                        <div className="font-medium">{alert.message}</div>
                        <div className="text-sm text-muted-foreground">
                          {alert.timestamp.toLocaleString()} • {alert.type.replace('_', ' ')}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline" className={getSeverityColor(alert.severity)}>
                        {alert.severity.toUpperCase()}
                      </Badge>
                      {!alert.acknowledged && (
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => handleAcknowledgeAlert(alert.id)}
                        >
                          Acknowledge
                        </Button>
                      )}
                      {alert.acknowledged && (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
