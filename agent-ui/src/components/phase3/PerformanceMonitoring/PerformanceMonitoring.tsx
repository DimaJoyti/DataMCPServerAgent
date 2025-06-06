'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import {
  Activity,
  Cpu,
  Database,
  HardDrive,
  Network,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3
} from 'lucide-react'

interface MetricData {
  timestamp: string
  value: number
}

interface PerformanceMetric {
  id: string
  name: string
  value: number
  unit: string
  trend: 'up' | 'down' | 'stable'
  status: 'good' | 'warning' | 'critical'
  history: MetricData[]
}

interface AgentPerformance {
  agentId: string
  agentName: string
  type: 'multimodal' | 'rag' | 'streaming'
  metrics: {
    responseTime: number
    throughput: number
    successRate: number
    cpuUsage: number
    memoryUsage: number
    errorRate: number
  }
  status: 'healthy' | 'warning' | 'critical'
}

interface PerformanceMonitoringProps {
  className?: string
}

const mockSystemMetrics: PerformanceMetric[] = [
  {
    id: 'cpu',
    name: 'CPU Usage',
    value: 45.2,
    unit: '%',
    trend: 'stable',
    status: 'good',
    history: Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (19 - i) * 60000).toISOString(),
      value: 40 + Math.random() * 20
    }))
  },
  {
    id: 'memory',
    name: 'Memory Usage',
    value: 68.7,
    unit: '%',
    trend: 'up',
    status: 'warning',
    history: Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (19 - i) * 60000).toISOString(),
      value: 60 + Math.random() * 15
    }))
  },
  {
    id: 'disk',
    name: 'Disk Usage',
    value: 23.4,
    unit: '%',
    trend: 'stable',
    status: 'good',
    history: Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (19 - i) * 60000).toISOString(),
      value: 20 + Math.random() * 10
    }))
  },
  {
    id: 'network',
    name: 'Network I/O',
    value: 156.8,
    unit: 'MB/s',
    trend: 'up',
    status: 'good',
    history: Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (19 - i) * 60000).toISOString(),
      value: 100 + Math.random() * 100
    }))
  }
]

const mockAgentPerformance: AgentPerformance[] = [
  {
    agentId: 'multimodal-1',
    agentName: 'Multimodal Agent 1',
    type: 'multimodal',
    metrics: {
      responseTime: 1.2,
      throughput: 45.6,
      successRate: 98.5,
      cpuUsage: 34.2,
      memoryUsage: 512.8,
      errorRate: 1.5
    },
    status: 'healthy'
  },
  {
    agentId: 'rag-1',
    agentName: 'RAG Agent 1',
    type: 'rag',
    metrics: {
      responseTime: 0.8,
      throughput: 67.3,
      successRate: 99.1,
      cpuUsage: 28.7,
      memoryUsage: 384.2,
      errorRate: 0.9
    },
    status: 'healthy'
  },
  {
    agentId: 'streaming-1',
    agentName: 'Streaming Agent 1',
    type: 'streaming',
    metrics: {
      responseTime: 0.3,
      throughput: 234.7,
      successRate: 97.8,
      cpuUsage: 52.1,
      memoryUsage: 768.4,
      errorRate: 2.2
    },
    status: 'warning'
  }
]

const getMetricIcon = (metricId: string) => {
  switch (metricId) {
    case 'cpu':
      return <Cpu className="h-4 w-4" />
    case 'memory':
      return <Database className="h-4 w-4" />
    case 'disk':
      return <HardDrive className="h-4 w-4" />
    case 'network':
      return <Network className="h-4 w-4" />
    default:
      return <Activity className="h-4 w-4" />
  }
}

const getTrendIcon = (trend: string) => {
  switch (trend) {
    case 'up':
      return <TrendingUp className="h-3 w-3 text-green-500" />
    case 'down':
      return <TrendingDown className="h-3 w-3 text-red-500" />
    default:
      return <Activity className="h-3 w-3 text-gray-500" />
  }
}

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'good':
    case 'healthy':
      return <CheckCircle className="h-4 w-4 text-green-500" />
    case 'warning':
      return <AlertTriangle className="h-4 w-4 text-yellow-500" />
    case 'critical':
      return <AlertTriangle className="h-4 w-4 text-red-500" />
    default:
      return <Clock className="h-4 w-4 text-gray-500" />
  }
}

const getStatusColor = (status: string) => {
  switch (status) {
    case 'good':
    case 'healthy':
      return 'bg-green-500'
    case 'warning':
      return 'bg-yellow-500'
    case 'critical':
      return 'bg-red-500'
    default:
      return 'bg-gray-500'
  }
}

export const PerformanceMonitoring: React.FC<PerformanceMonitoringProps> = ({ className }) => {
  const [systemMetrics, setSystemMetrics] = useState<PerformanceMetric[]>(mockSystemMetrics)
  const [agentPerformance, setAgentPerformance] = useState<AgentPerformance[]>(mockAgentPerformance)

  useEffect(() => {
    // Simulate real-time updates
    const interval = setInterval(() => {
      setSystemMetrics(prevMetrics =>
        prevMetrics.map(metric => ({
          ...metric,
          value: Math.max(0, Math.min(100, metric.value + (Math.random() - 0.5) * 5)),
          history: [
            ...metric.history.slice(1),
            {
              timestamp: new Date().toISOString(),
              value: Math.max(0, Math.min(100, metric.value + (Math.random() - 0.5) * 5))
            }
          ]
        }))
      )

      setAgentPerformance(prevPerformance =>
        prevPerformance.map(agent => ({
          ...agent,
          metrics: {
            ...agent.metrics,
            responseTime: Math.max(0.1, agent.metrics.responseTime + (Math.random() - 0.5) * 0.2),
            throughput: Math.max(0, agent.metrics.throughput + (Math.random() - 0.5) * 10),
            cpuUsage: Math.max(0, Math.min(100, agent.metrics.cpuUsage + (Math.random() - 0.5) * 5)),
            memoryUsage: Math.max(0, agent.metrics.memoryUsage + (Math.random() - 0.5) * 50)
          }
        }))
      )
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Performance Monitoring</h1>
          <p className="text-muted-foreground">
            Real-time system and agent performance metrics
          </p>
        </div>
        <Button>
          <BarChart3 className="mr-2 h-4 w-4" />
          View Reports
        </Button>
      </div>

      {/* System Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {systemMetrics.map((metric) => (
          <Card key={metric.id}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{metric.name}</CardTitle>
              {getMetricIcon(metric.id)}
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between mb-2">
                <div className="text-2xl font-bold">
                  {metric.value.toFixed(1)}{metric.unit}
                </div>
                <div className="flex items-center space-x-1">
                  {getTrendIcon(metric.trend)}
                  {getStatusIcon(metric.status)}
                </div>
              </div>
              <Progress value={metric.value} className="h-2" />
              <p className="text-xs text-muted-foreground mt-1">
                Last updated: {new Date().toLocaleTimeString()}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Agent Performance */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="detailed">Detailed Metrics</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4">
            {agentPerformance.map((agent) => (
              <Card key={agent.agentId}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg">{agent.agentName}</CardTitle>
                      <CardDescription>
                        {agent.type.charAt(0).toUpperCase() + agent.type.slice(1)} Agent
                      </CardDescription>
                    </div>
                    <Badge variant={agent.status === 'healthy' ? 'default' : 'secondary'}>
                      <div className={`w-2 h-2 rounded-full mr-2 ${getStatusColor(agent.status)}`} />
                      {agent.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Response Time</p>
                      <p className="font-medium">{agent.metrics.responseTime.toFixed(1)}s</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Throughput</p>
                      <p className="font-medium">{agent.metrics.throughput.toFixed(1)}/min</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Success Rate</p>
                      <p className="font-medium">{agent.metrics.successRate.toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">CPU Usage</p>
                      <div className="flex items-center space-x-2">
                        <Progress value={agent.metrics.cpuUsage} className="flex-1 h-2" />
                        <span className="text-xs">{agent.metrics.cpuUsage.toFixed(1)}%</span>
                      </div>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Memory</p>
                      <p className="font-medium">{agent.metrics.memoryUsage.toFixed(0)}MB</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Error Rate</p>
                      <p className="font-medium">{agent.metrics.errorRate.toFixed(1)}%</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="detailed">
          <div className="text-center py-8">
            <BarChart3 className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">Detailed Metrics</h3>
            <p className="text-muted-foreground">
              Comprehensive performance analytics and historical data
            </p>
          </div>
        </TabsContent>

        <TabsContent value="alerts">
          <div className="text-center py-8">
            <AlertTriangle className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">Performance Alerts</h3>
            <p className="text-muted-foreground">
              Configure and monitor performance thresholds and alerts
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
