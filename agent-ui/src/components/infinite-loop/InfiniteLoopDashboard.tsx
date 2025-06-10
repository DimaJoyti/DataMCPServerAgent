'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { 
  Infinity, 
  Play, 
  Pause, 
  Square, 
  Settings, 
  Activity, 
  BarChart3, 
  Users, 
  Zap,
  AlertCircle,
  CheckCircle,
  Clock,
  TrendingUp
} from 'lucide-react'
import { useInfiniteLoopStore } from '@/store'
import { useInfiniteLoopWebSocket } from '@/hooks/useInfiniteLoopWebSocket'
import { ConfigurationPanel } from './ConfigurationPanel'
import { ExecutionMonitor } from './ExecutionMonitor'
import { AnalyticsDashboard } from './AnalyticsDashboard'
import { AgentPoolViewer } from './AgentPoolViewer'

export const InfiniteLoopDashboard: React.FC = () => {
  const {
    currentSession,
    uiState,
    setUIState,
    isConnected,
    errors
  } = useInfiniteLoopStore()

  const [activeTab, setActiveTab] = useState<string>('overview')

  // Initialize WebSocket connection if there's an active session
  useInfiniteLoopWebSocket(currentSession?.sessionId)

  useEffect(() => {
    // Initialize dashboard
    setUIState({ selectedView: 'dashboard' })
  }, [setUIState])

  const handleStartExecution = () => {
    setUIState({ selectedView: 'configuration' })
  }

  const handleStopExecution = () => {
    // Implementation will be added
    console.log('Stop execution')
  }

  const handlePauseExecution = () => {
    // Implementation will be added
    console.log('Pause execution')
  }

  const getExecutionStatus = () => {
    if (!currentSession) return 'idle'
    return currentSession.executionState.isRunning ? 'running' : 'stopped'
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500'
      case 'paused': return 'bg-yellow-500'
      case 'stopped': return 'bg-red-500'
      case 'error': return 'bg-red-600'
      default: return 'bg-gray-500'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Play className="h-4 w-4" />
      case 'paused': return <Pause className="h-4 w-4" />
      case 'stopped': return <Square className="h-4 w-4" />
      case 'error': return <AlertCircle className="h-4 w-4" />
      default: return <Clock className="h-4 w-4" />
    }
  }

  const executionStatus = getExecutionStatus()

  return (
    <div className="h-full bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      <div className="h-full flex flex-col">
        {/* Header */}
        <div className="border-b border-border/50 backdrop-blur-sm bg-white/80 dark:bg-slate-900/80 p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center shadow-lg">
                <Infinity className="h-7 w-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-300 bg-clip-text text-transparent">
                  Infinite Agentic Loop
                </h1>
                <p className="text-sm text-muted-foreground">
                  Advanced AI Agent Orchestration System
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Connection Status */}
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
                <span className="text-sm text-muted-foreground">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>

              {/* Execution Status */}
              {currentSession && (
                <Badge variant="outline" className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${getStatusColor(executionStatus)}`} />
                  {getStatusIcon(executionStatus)}
                  <span className="capitalize">{executionStatus}</span>
                </Badge>
              )}

              {/* Control Buttons */}
              <div className="flex items-center space-x-2">
                {!currentSession || !currentSession.executionState.isRunning ? (
                  <Button
                    onClick={handleStartExecution}
                    className="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700"
                  >
                    <Play className="mr-2 h-4 w-4" />
                    Start Loop
                  </Button>
                ) : (
                  <>
                    <Button
                      variant="outline"
                      onClick={handlePauseExecution}
                      className="border-yellow-500 text-yellow-600 hover:bg-yellow-50"
                    >
                      <Pause className="mr-2 h-4 w-4" />
                      Pause
                    </Button>
                    <Button
                      variant="outline"
                      onClick={handleStopExecution}
                      className="border-red-500 text-red-600 hover:bg-red-50"
                    >
                      <Square className="mr-2 h-4 w-4" />
                      Stop
                    </Button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-6">
          {uiState.selectedView === 'configuration' ? (
            <ConfigurationPanel />
          ) : (
            <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full">
              <TabsList className="grid w-full grid-cols-4 mb-6">
                <TabsTrigger value="overview" className="flex items-center space-x-2">
                  <Activity className="h-4 w-4" />
                  <span>Overview</span>
                </TabsTrigger>
                <TabsTrigger value="execution" className="flex items-center space-x-2">
                  <Zap className="h-4 w-4" />
                  <span>Execution</span>
                </TabsTrigger>
                <TabsTrigger value="agents" className="flex items-center space-x-2">
                  <Users className="h-4 w-4" />
                  <span>Agents</span>
                </TabsTrigger>
                <TabsTrigger value="analytics" className="flex items-center space-x-2">
                  <BarChart3 className="h-4 w-4" />
                  <span>Analytics</span>
                </TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="h-full">
                <OverviewPanel currentSession={currentSession} errors={errors} />
              </TabsContent>

              <TabsContent value="execution" className="h-full">
                <ExecutionMonitor />
              </TabsContent>

              <TabsContent value="agents" className="h-full">
                <AgentPoolViewer />
              </TabsContent>

              <TabsContent value="analytics" className="h-full">
                <AnalyticsDashboard />
              </TabsContent>
            </Tabs>
          )}
        </div>
      </div>
    </div>
  )
}

// Overview Panel Component
const OverviewPanel: React.FC<{
  currentSession: any
  errors: any[]
}> = ({ currentSession, errors }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 h-full">
      {/* Session Info */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <CheckCircle className="h-5 w-5 text-green-500" />
            <span>Current Session</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {currentSession ? (
            <div className="space-y-2">
              <p className="text-sm"><strong>ID:</strong> {currentSession.sessionId}</p>
              <p className="text-sm"><strong>Started:</strong> {new Date(currentSession.createdAt).toLocaleString()}</p>
              <p className="text-sm"><strong>Iterations:</strong> {currentSession.executionState.totalIterations}</p>
              <p className="text-sm"><strong>Waves:</strong> {currentSession.executionState.currentWave}</p>
            </div>
          ) : (
            <p className="text-muted-foreground">No active session</p>
          )}
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5 text-blue-500" />
            <span>Performance</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {currentSession ? (
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Success Rate</span>
                  <span>{(currentSession.performanceMetrics.successRate * 100).toFixed(1)}%</span>
                </div>
                <Progress value={currentSession.performanceMetrics.successRate * 100} />
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Quality Score</span>
                  <span>{(currentSession.qualityMetrics.overallScore * 100).toFixed(1)}%</span>
                </div>
                <Progress value={currentSession.qualityMetrics.overallScore * 100} />
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground">No metrics available</p>
          )}
        </CardContent>
      </Card>

      {/* System Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-purple-500" />
            <span>System Status</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm">Errors</span>
              <Badge variant={errors.length > 0 ? "destructive" : "secondary"}>
                {errors.length}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-sm">Active Agents</span>
              <Badge variant="outline">
                {currentSession?.agents?.filter((a: any) => a.status === 'running').length || 0}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
