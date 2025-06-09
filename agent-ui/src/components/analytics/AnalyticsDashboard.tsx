'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown, 
  Users, 
  MessageSquare, 
  Clock, 
  Star, 
  AlertTriangle,
  CheckCircle,
  Activity,
  Brain,
  Target,
  Zap,
  ArrowUp,
  ArrowDown,
  Minus,
  RefreshCw
} from 'lucide-react'

// Types
interface MetricCard {
  title: string
  value: string | number
  change: number
  changeType: 'increase' | 'decrease' | 'neutral'
  icon: React.ComponentType<any>
  description?: string
}

interface ChartData {
  name: string
  value: number
  timestamp: string
}

interface Alert {
  id: string
  type: 'warning' | 'error' | 'info'
  title: string
  message: string
  timestamp: string
}

interface AnalyticsDashboardProps {
  agentId?: string
  timeRange: 'hour' | 'day' | 'week' | 'month'
  onTimeRangeChange: (range: 'hour' | 'day' | 'week' | 'month') => void
}

export const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({
  agentId,
  timeRange,
  onTimeRangeChange
}) => {
  const [metrics, setMetrics] = useState<MetricCard[]>([])
  const [chartData, setChartData] = useState<ChartData[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())

  // Mock data - would be fetched from API
  useEffect(() => {
    const fetchAnalytics = async () => {
      setIsLoading(true)
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Mock metrics data
      const mockMetrics: MetricCard[] = [
        {
          title: 'Active Conversations',
          value: 42,
          change: 12.5,
          changeType: 'increase',
          icon: MessageSquare,
          description: 'Currently active conversations'
        },
        {
          title: 'Average Satisfaction',
          value: '4.2/5',
          change: 8.3,
          changeType: 'increase',
          icon: Star,
          description: 'User satisfaction rating'
        },
        {
          title: 'Response Time',
          value: '1.8s',
          change: -15.2,
          changeType: 'decrease',
          icon: Clock,
          description: 'Average response time'
        },
        {
          title: 'Resolution Rate',
          value: '87%',
          change: 5.1,
          changeType: 'increase',
          icon: CheckCircle,
          description: 'Conversations resolved successfully'
        },
        {
          title: 'Total Users',
          value: 1247,
          change: 23.4,
          changeType: 'increase',
          icon: Users,
          description: 'Unique users served'
        },
        {
          title: 'Escalation Rate',
          value: '3.2%',
          change: -8.7,
          changeType: 'decrease',
          icon: AlertTriangle,
          description: 'Conversations escalated to humans'
        }
      ]
      
      // Mock chart data
      const mockChartData: ChartData[] = Array.from({ length: 24 }, (_, i) => ({
        name: `${i}:00`,
        value: Math.floor(Math.random() * 100) + 20,
        timestamp: new Date(Date.now() - (23 - i) * 60 * 60 * 1000).toISOString()
      }))
      
      // Mock alerts
      const mockAlerts: Alert[] = [
        {
          id: '1',
          type: 'warning',
          title: 'High Response Time',
          message: 'Average response time exceeded 3 seconds in the last hour',
          timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString()
        },
        {
          id: '2',
          type: 'info',
          title: 'Peak Traffic',
          message: 'Conversation volume is 40% higher than usual',
          timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString()
        }
      ]
      
      setMetrics(mockMetrics)
      setChartData(mockChartData)
      setAlerts(mockAlerts)
      setLastUpdated(new Date())
      setIsLoading(false)
    }
    
    fetchAnalytics()
    
    // Set up auto-refresh
    const interval = setInterval(fetchAnalytics, 30000) // Refresh every 30 seconds
    
    return () => clearInterval(interval)
  }, [agentId, timeRange])

  const getChangeIcon = (changeType: 'increase' | 'decrease' | 'neutral') => {
    switch (changeType) {
      case 'increase':
        return <ArrowUp className="h-4 w-4 text-green-500" />
      case 'decrease':
        return <ArrowDown className="h-4 w-4 text-red-500" />
      default:
        return <Minus className="h-4 w-4 text-gray-500" />
    }
  }

  const getChangeColor = (changeType: 'increase' | 'decrease' | 'neutral') => {
    switch (changeType) {
      case 'increase':
        return 'text-green-600'
      case 'decrease':
        return 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }

  const getAlertIcon = (type: 'warning' | 'error' | 'info') => {
    switch (type) {
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-500" />
      default:
        return <Activity className="h-4 w-4 text-blue-500" />
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin text-blue-500" />
          <span className="ml-2 text-lg">Loading analytics...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Analytics Dashboard</h1>
          <p className="text-muted-foreground">
            Real-time insights and performance metrics
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* Time Range Selector */}
          <div className="flex items-center space-x-2">
            {(['hour', 'day', 'week', 'month'] as const).map((range) => (
              <Button
                key={range}
                variant={timeRange === range ? 'default' : 'outline'}
                size="sm"
                onClick={() => onTimeRangeChange(range)}
              >
                {range.charAt(0).toUpperCase() + range.slice(1)}
              </Button>
            ))}
          </div>
          
          {/* Last Updated */}
          <div className="text-sm text-muted-foreground">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertTriangle className="mr-2 h-5 w-5" />
              Active Alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {alerts.map((alert) => (
                <div
                  key={alert.id}
                  className="flex items-start space-x-3 p-3 border rounded-lg"
                >
                  {getAlertIcon(alert.type)}
                  <div className="flex-1">
                    <h4 className="font-medium">{alert.title}</h4>
                    <p className="text-sm text-muted-foreground">{alert.message}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {formatTimestamp(alert.timestamp)}
                    </p>
                  </div>
                  <Button variant="outline" size="sm">
                    Dismiss
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metrics.map((metric, index) => {
          const Icon = metric.icon
          return (
            <Card key={index}>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
                      <Icon className="h-5 w-5 text-blue-600" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">
                        {metric.title}
                      </p>
                      <p className="text-2xl font-bold">{metric.value}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-1">
                    {getChangeIcon(metric.changeType)}
                    <span className={`text-sm font-medium ${getChangeColor(metric.changeType)}`}>
                      {Math.abs(metric.change)}%
                    </span>
                  </div>
                </div>
                
                {metric.description && (
                  <p className="text-xs text-muted-foreground mt-2">
                    {metric.description}
                  </p>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Detailed Analytics Tabs */}
      <Tabs defaultValue="performance" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="performance" className="flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Performance</span>
          </TabsTrigger>
          <TabsTrigger value="conversations" className="flex items-center space-x-2">
            <MessageSquare className="h-4 w-4" />
            <span>Conversations</span>
          </TabsTrigger>
          <TabsTrigger value="learning" className="flex items-center space-x-2">
            <Brain className="h-4 w-4" />
            <span>Learning</span>
          </TabsTrigger>
          <TabsTrigger value="experiments" className="flex items-center space-x-2">
            <Target className="h-4 w-4" />
            <span>A/B Tests</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="performance" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Response Time Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Response Time Trend</CardTitle>
                <CardDescription>
                  Average response time over the last {timeRange}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center border-2 border-dashed border-gray-200 rounded-lg">
                  <div className="text-center">
                    <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-500">Chart visualization would go here</p>
                    <p className="text-sm text-gray-400">
                      Integration with charting library (Chart.js, Recharts, etc.)
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Satisfaction Trend */}
            <Card>
              <CardHeader>
                <CardTitle>Satisfaction Trend</CardTitle>
                <CardDescription>
                  User satisfaction ratings over time
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center border-2 border-dashed border-gray-200 rounded-lg">
                  <div className="text-center">
                    <Star className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-500">Satisfaction chart would go here</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Performance Insights */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Insights</CardTitle>
              <CardDescription>
                AI-generated insights about agent performance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start space-x-3 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <TrendingUp className="h-5 w-5 text-green-600 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-green-800">Improved Response Time</h4>
                    <p className="text-sm text-green-700">
                      Response time has improved by 15% compared to last week. 
                      The optimization of knowledge retrieval is showing positive results.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <Zap className="h-5 w-5 text-blue-600 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-blue-800">Peak Performance Hours</h4>
                    <p className="text-sm text-blue-700">
                      Agent performs best between 10 AM - 2 PM with 92% resolution rate.
                      Consider adjusting staffing during these hours.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="conversations" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Conversation Volume */}
            <Card>
              <CardHeader>
                <CardTitle>Conversation Volume</CardTitle>
                <CardDescription>
                  Number of conversations over time
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center border-2 border-dashed border-gray-200 rounded-lg">
                  <div className="text-center">
                    <MessageSquare className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-500">Volume chart would go here</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Top Topics */}
            <Card>
              <CardHeader>
                <CardTitle>Top Discussion Topics</CardTitle>
                <CardDescription>
                  Most frequently discussed topics
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { topic: 'Product Information', count: 156, percentage: 35 },
                    { topic: 'Technical Support', count: 98, percentage: 22 },
                    { topic: 'Billing Questions', count: 87, percentage: 19 },
                    { topic: 'Account Issues', count: 65, percentage: 15 },
                    { topic: 'General Inquiry', count: 42, percentage: 9 },
                  ].map((item, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium">{item.topic}</span>
                          <span className="text-sm text-muted-foreground">{item.count}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-500 h-2 rounded-full" 
                            style={{ width: `${item.percentage}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="learning" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Learning Insights</CardTitle>
              <CardDescription>
                AI-generated insights from conversation analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">Response Pattern Optimization</h4>
                    <Badge variant="outline">High Impact</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Users respond 23% better to empathetic tone when expressing frustration.
                    Consider adjusting personality settings for negative sentiment detection.
                  </p>
                  <div className="flex items-center space-x-2">
                    <Button size="sm">Apply Recommendation</Button>
                    <Button size="sm" variant="outline">Learn More</Button>
                  </div>
                </div>
                
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">Knowledge Gap Identified</h4>
                    <Badge variant="outline">Medium Impact</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    15% of escalations are related to "refund policy" questions. 
                    Adding comprehensive refund knowledge could reduce escalations.
                  </p>
                  <div className="flex items-center space-x-2">
                    <Button size="sm">Add Knowledge</Button>
                    <Button size="sm" variant="outline">View Details</Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="experiments" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Active A/B Tests</CardTitle>
              <CardDescription>
                Currently running experiments and their results
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">Personality Tone Test</h4>
                    <Badge>Running</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Testing formal vs. casual tone for customer support interactions
                  </p>
                  <div className="grid grid-cols-2 gap-4 mb-3">
                    <div className="text-center p-2 bg-gray-50 rounded">
                      <p className="text-sm font-medium">Control (Formal)</p>
                      <p className="text-lg font-bold">4.1/5</p>
                      <p className="text-xs text-muted-foreground">245 participants</p>
                    </div>
                    <div className="text-center p-2 bg-blue-50 rounded">
                      <p className="text-sm font-medium">Test (Casual)</p>
                      <p className="text-lg font-bold">4.3/5</p>
                      <p className="text-xs text-muted-foreground">238 participants</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button size="sm">View Details</Button>
                    <Button size="sm" variant="outline">Stop Test</Button>
                  </div>
                </div>
                
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">Response Length Test</h4>
                    <Badge variant="secondary">Completed</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Testing short vs. detailed responses for technical questions
                  </p>
                  <div className="flex items-center space-x-2 text-sm">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span>Winner: Detailed responses (+12% satisfaction)</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
