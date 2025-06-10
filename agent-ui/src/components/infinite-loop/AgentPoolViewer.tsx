'use client'

import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  Users, 
  User, 
  Activity, 
  CheckCircle, 
  AlertCircle, 
  Clock,
  Zap,
  TrendingUp
} from 'lucide-react'
import { useInfiniteLoopStore } from '@/store'

export const AgentPoolViewer: React.FC = () => {
  const { currentSession } = useInfiniteLoopStore()

  if (!currentSession) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <Users className="h-12 w-12 mx-auto text-muted-foreground" />
          <div>
            <h3 className="text-lg font-semibold">No Active Session</h3>
            <p className="text-muted-foreground">Start an infinite loop to view agent pool</p>
          </div>
        </div>
      </div>
    )
  }

  const { agents = [], config } = currentSession

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500'
      case 'idle': return 'bg-blue-500'
      case 'completed': return 'bg-gray-500'
      case 'error': return 'bg-red-500'
      default: return 'bg-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Activity className="h-4 w-4" />
      case 'idle': return <Clock className="h-4 w-4" />
      case 'completed': return <CheckCircle className="h-4 w-4" />
      case 'error': return <AlertCircle className="h-4 w-4" />
      default: return <User className="h-4 w-4" />
    }
  }

  const activeAgents = agents.filter(agent => agent.status === 'running')
  const idleAgents = agents.filter(agent => agent.status === 'idle')
  const completedAgents = agents.filter(agent => agent.status === 'completed')
  const errorAgents = agents.filter(agent => agent.status === 'error')

  return (
    <div className="space-y-6">
      {/* Pool Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Agents</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{agents.length}</div>
            <p className="text-sm text-muted-foreground">
              Max: {config.maxParallelAgents}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Active</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <div className="text-2xl font-bold">{activeAgents.length}</div>
            </div>
            <p className="text-sm text-muted-foreground">Running tasks</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Idle</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full" />
              <div className="text-2xl font-bold">{idleAgents.length}</div>
            </div>
            <p className="text-sm text-muted-foreground">Available</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Errors</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-red-500 rounded-full" />
              <div className="text-2xl font-bold">{errorAgents.length}</div>
            </div>
            <p className="text-sm text-muted-foreground">Failed agents</p>
          </CardContent>
        </Card>
      </div>

      {/* Agent List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Users className="h-5 w-5" />
            <span>Agent Pool</span>
          </CardTitle>
          <CardDescription>
            Individual agent status and performance metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          {agents.length > 0 ? (
            <div className="space-y-4">
              {agents.map((agent) => (
                <div key={agent.agentId} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${getStatusColor(agent.status)}`} />
                      <div>
                        <div className="font-medium">Agent {agent.agentId}</div>
                        <div className="text-sm text-muted-foreground">
                          {agent.currentTask || 'No active task'}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline" className="flex items-center space-x-1">
                        {getStatusIcon(agent.status)}
                        <span className="capitalize">{agent.status}</span>
                      </Badge>
                      {agent.iterationNumber && (
                        <Badge variant="secondary">
                          Iteration {agent.iterationNumber}
                        </Badge>
                      )}
                    </div>
                  </div>

                  {/* Progress Bar */}
                  <div className="mb-3">
                    <div className="flex justify-between text-sm mb-1">
                      <span>Progress</span>
                      <span>{agent.progress}%</span>
                    </div>
                    <Progress value={agent.progress} />
                  </div>

                  {/* Metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Total Tasks:</span>
                      <div className="font-medium">{agent.metrics.totalTasks}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Success Rate:</span>
                      <div className="font-medium">{(agent.metrics.successRate * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Avg Time:</span>
                      <div className="font-medium">{agent.metrics.averageExecutionTime.toFixed(1)}s</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Quality:</span>
                      <div className="font-medium">{(agent.metrics.qualityScore * 100).toFixed(1)}%</div>
                    </div>
                  </div>

                  {/* Start Time */}
                  {agent.startTime && (
                    <div className="mt-2 text-xs text-muted-foreground">
                      Started: {new Date(agent.startTime).toLocaleString()}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Users className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
              <p className="text-muted-foreground">No agents in pool</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Pool Performance */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5" />
            <span>Pool Performance</span>
          </CardTitle>
          <CardDescription>
            Aggregate performance metrics across all agents
          </CardDescription>
        </CardHeader>
        <CardContent>
          {agents.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Overall Success Rate</span>
                  <span>
                    {agents.length > 0 
                      ? ((agents.reduce((sum, agent) => sum + agent.metrics.successRate, 0) / agents.length) * 100).toFixed(1)
                      : 0
                    }%
                  </span>
                </div>
                <Progress 
                  value={agents.length > 0 
                    ? (agents.reduce((sum, agent) => sum + agent.metrics.successRate, 0) / agents.length) * 100
                    : 0
                  } 
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Average Quality</span>
                  <span>
                    {agents.length > 0 
                      ? ((agents.reduce((sum, agent) => sum + agent.metrics.qualityScore, 0) / agents.length) * 100).toFixed(1)
                      : 0
                    }%
                  </span>
                </div>
                <Progress 
                  value={agents.length > 0 
                    ? (agents.reduce((sum, agent) => sum + agent.metrics.qualityScore, 0) / agents.length) * 100
                    : 0
                  } 
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Pool Utilization</span>
                  <span>{((agents.length / config.maxParallelAgents) * 100).toFixed(1)}%</span>
                </div>
                <Progress value={(agents.length / config.maxParallelAgents) * 100} />
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <TrendingUp className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
              <p className="text-muted-foreground">No performance data available</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
