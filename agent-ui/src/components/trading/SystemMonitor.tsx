'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import {
  Activity,
  Server,
  Database,
  Wifi,
  Zap,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Settings
} from 'lucide-react'

interface SystemComponent {
  name: string
  status: 'online' | 'degraded' | 'offline'
  uptime: number
  latency?: number
  throughput?: number
  errorRate?: number
  lastUpdate: Date
}

interface SystemMetric {
  name: string
  value: number
  unit: string
  status: 'good' | 'warning' | 'critical'
  threshold: number
}

interface SystemAlert {
  id: string
  component: string
  severity: 'info' | 'warning' | 'error'
  message: string
  timestamp: Date
  resolved: boolean
}

export const SystemMonitor: React.FC = () => {
  const [systemComponents, setSystemComponents] = useState<SystemComponent[]>([
    {
      name: 'Order Management System',
      status: 'online',
      uptime: 99.98,
      latency: 0.8,
      throughput: 12500,
      errorRate: 0.02,
      lastUpdate: new Date()
    },
    {
      name: 'Market Data Feed',
      status: 'online',
      uptime: 99.95,
      latency: 2.1,
      throughput: 50000,
      errorRate: 0.01,
      lastUpdate: new Date()
    },
    {
      name: 'Risk Management Engine',
      status: 'online',
      uptime: 99.99,
      latency: 1.2,
      throughput: 8000,
      errorRate: 0.00,
      lastUpdate: new Date()
    },
    {
      name: 'Execution Gateway',
      status: 'degraded',
      uptime: 98.75,
      latency: 5.8,
      throughput: 3200,
      errorRate: 1.25,
      lastUpdate: new Date()
    },
    {
      name: 'Database Cluster',
      status: 'online',
      uptime: 99.97,
      latency: 3.2,
      errorRate: 0.05,
      lastUpdate: new Date()
    },
    {
      name: 'Analytics Engine',
      status: 'online',
      uptime: 99.92,
      latency: 15.5,
      throughput: 1500,
      errorRate: 0.08,
      lastUpdate: new Date()
    }
  ])

  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>([
    {
      name: 'CPU Usage',
      value: 45.2,
      unit: '%',
      status: 'good',
      threshold: 80
    },
    {
      name: 'Memory Usage',
      value: 68.5,
      unit: '%',
      status: 'warning',
      threshold: 85
    },
    {
      name: 'Disk I/O',
      value: 25.8,
      unit: '%',
      status: 'good',
      threshold: 70
    },
    {
      name: 'Network Latency',
      value: 1.2,
      unit: 'ms',
      status: 'good',
      threshold: 5
    },
    {
      name: 'Orders/Second',
      value: 8750,
      unit: 'ops',
      status: 'good',
      threshold: 10000
    },
    {
      name: 'Error Rate',
      value: 0.15,
      unit: '%',
      status: 'good',
      threshold: 1
    }
  ])

  const [systemAlerts, setSystemAlerts] = useState<SystemAlert[]>([
    {
      id: 'SYS-001',
      component: 'Execution Gateway',
      severity: 'warning',
      message: 'Increased latency detected on NYSE connection',
      timestamp: new Date(Date.now() - 300000),
      resolved: false
    },
    {
      id: 'SYS-002',
      component: 'Market Data Feed',
      severity: 'info',
      message: 'Scheduled maintenance completed successfully',
      timestamp: new Date(Date.now() - 600000),
      resolved: true
    },
    {
      id: 'SYS-003',
      component: 'Database Cluster',
      severity: 'error',
      message: 'Connection pool exhaustion on primary node',
      timestamp: new Date(Date.now() - 900000),
      resolved: true
    }
  ])

  useEffect(() => {
    // Simulate real-time system updates
    const interval = setInterval(() => {
      setSystemComponents(prev => prev.map(component => ({
        ...component,
        latency: component.latency ? component.latency + (Math.random() - 0.5) * 2 : undefined,
        throughput: component.throughput ? component.throughput + Math.floor((Math.random() - 0.5) * 1000) : undefined,
        errorRate: component.errorRate ? Math.max(0, component.errorRate + (Math.random() - 0.5) * 0.1) : undefined,
        lastUpdate: new Date()
      })))

      setSystemMetrics(prev => prev.map(metric => {
        const change = (Math.random() - 0.5) * metric.value * 0.1
        const newValue = Math.max(0, metric.value + change)
        let status: 'good' | 'warning' | 'critical' = 'good'
        
        if (metric.name === 'Error Rate') {
          if (newValue > metric.threshold * 0.8) status = 'critical'
          else if (newValue > metric.threshold * 0.5) status = 'warning'
        } else {
          if (newValue > metric.threshold * 0.9) status = 'critical'
          else if (newValue > metric.threshold * 0.7) status = 'warning'
        }

        return { ...metric, value: newValue, status }
      }))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
      case 'good': return 'text-green-500'
      case 'degraded':
      case 'warning': return 'text-yellow-500'
      case 'offline':
      case 'critical': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
      case 'good': return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'degraded':
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      case 'offline':
      case 'critical': return <XCircle className="h-4 w-4 text-red-500" />
      default: return <Activity className="h-4 w-4 text-gray-500" />
    }
  }

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'info': return <Activity className="h-4 w-4 text-blue-500" />
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      case 'error': return <XCircle className="h-4 w-4 text-red-500" />
      default: return <Activity className="h-4 w-4 text-gray-500" />
    }
  }

  const formatValue = (value: number, unit: string) => {
    if (unit === 'ops') {
      return value.toLocaleString()
    }
    return `${value.toFixed(unit === 'ms' ? 1 : 0)}${unit}`
  }

  return (
    <div className="h-full space-y-6 p-6">
      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg font-bold flex items-center space-x-2">
              <Server className="h-5 w-5" />
              <span>System Health</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className="text-4xl font-bold text-green-500">
                {systemComponents.filter(c => c.status === 'online').length}
              </div>
              <div className="text-sm text-muted-foreground mb-4">
                of {systemComponents.length} components online
              </div>
              <div className="space-y-1">
                <div className="text-xs">
                  <span className="text-green-500">Online: </span>
                  {systemComponents.filter(c => c.status === 'online').length}
                </div>
                <div className="text-xs">
                  <span className="text-yellow-500">Degraded: </span>
                  {systemComponents.filter(c => c.status === 'degraded').length}
                </div>
                <div className="text-xs">
                  <span className="text-red-500">Offline: </span>
                  {systemComponents.filter(c => c.status === 'offline').length}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg font-bold flex items-center space-x-2">
              <Zap className="h-5 w-5" />
              <span>Performance</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className="text-4xl font-bold text-blue-500">
                {systemMetrics.find(m => m.name === 'Orders/Second')?.value.toLocaleString()}
              </div>
              <div className="text-sm text-muted-foreground mb-4">orders per second</div>
              <div className="space-y-1">
                <div className="text-xs">
                  Avg Latency: {systemMetrics.find(m => m.name === 'Network Latency')?.value.toFixed(1)}ms
                </div>
                <div className="text-xs">
                  Error Rate: {systemMetrics.find(m => m.name === 'Error Rate')?.value.toFixed(2)}%
                </div>
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
              <div className="text-4xl font-bold text-yellow-500">
                {systemAlerts.filter(a => !a.resolved).length}
              </div>
              <div className="text-sm text-muted-foreground mb-4">unresolved alerts</div>
              <div className="space-y-1">
                <div className="text-xs">
                  <span className="text-red-500">Error: </span>
                  {systemAlerts.filter(a => a.severity === 'error' && !a.resolved).length}
                </div>
                <div className="text-xs">
                  <span className="text-yellow-500">Warning: </span>
                  {systemAlerts.filter(a => a.severity === 'warning' && !a.resolved).length}
                </div>
                <div className="text-xs">
                  <span className="text-blue-500">Info: </span>
                  {systemAlerts.filter(a => a.severity === 'info' && !a.resolved).length}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* System Components */}
      <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-xl font-bold">System Components</CardTitle>
              <CardDescription>Real-time status of all trading system components</CardDescription>
            </div>
            <Button variant="outline" size="sm">
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {systemComponents.map((component, index) => (
              <div key={index} className="border border-border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(component.status)}
                    <div>
                      <div className="font-medium">{component.name}</div>
                      <div className="text-sm text-muted-foreground">
                        Uptime: {component.uptime.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                  <Badge variant={component.status === 'online' ? 'default' : component.status === 'degraded' ? 'secondary' : 'destructive'}>
                    {component.status.toUpperCase()}
                  </Badge>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  {component.latency && (
                    <div>
                      <span className="text-muted-foreground">Latency:</span>
                      <span className="ml-2 font-medium">{component.latency.toFixed(1)}ms</span>
                    </div>
                  )}
                  {component.throughput && (
                    <div>
                      <span className="text-muted-foreground">Throughput:</span>
                      <span className="ml-2 font-medium">{component.throughput.toLocaleString()}/s</span>
                    </div>
                  )}
                  {component.errorRate !== undefined && (
                    <div>
                      <span className="text-muted-foreground">Error Rate:</span>
                      <span className="ml-2 font-medium">{component.errorRate.toFixed(2)}%</span>
                    </div>
                  )}
                  <div>
                    <span className="text-muted-foreground">Updated:</span>
                    <span className="ml-2 font-medium">{component.lastUpdate.toLocaleTimeString()}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* System Metrics */}
      <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
        <CardHeader>
          <CardTitle className="text-xl font-bold">System Metrics</CardTitle>
          <CardDescription>Key performance indicators and resource utilization</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {systemMetrics.map((metric, index) => (
              <div key={index} className="border border-border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{metric.name}</span>
                  {getStatusIcon(metric.status)}
                </div>
                <div className="space-y-2">
                  <div className={`text-xl font-bold ${getStatusColor(metric.status)}`}>
                    {formatValue(metric.value, metric.unit)}
                  </div>
                  <Progress 
                    value={(metric.value / metric.threshold) * 100} 
                    className="h-2"
                  />
                  <div className="text-xs text-muted-foreground">
                    Threshold: {formatValue(metric.threshold, metric.unit)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* System Alerts */}
      <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
        <CardHeader>
          <CardTitle className="text-xl font-bold">System Alerts</CardTitle>
          <CardDescription>Recent system notifications and alerts</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {systemAlerts.map((alert) => (
              <div 
                key={alert.id} 
                className={`border rounded-lg p-4 ${
                  alert.resolved ? 'opacity-60 bg-muted/20' : 'bg-background'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getSeverityIcon(alert.severity)}
                    <div>
                      <div className="font-medium">{alert.message}</div>
                      <div className="text-sm text-muted-foreground">
                        {alert.component} • {alert.timestamp.toLocaleString()}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline">
                      {alert.severity.toUpperCase()}
                    </Badge>
                    {alert.resolved && (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
