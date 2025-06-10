'use client'

import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  Activity, 
  Clock, 
  Zap, 
  CheckCircle, 
  AlertCircle, 
  Play,
  Pause,
  Square
} from 'lucide-react'
import { useInfiniteLoopStore } from '@/store'

export const ExecutionMonitor: React.FC = () => {
  const { currentSession } = useInfiniteLoopStore()

  if (!currentSession) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <Activity className="h-12 w-12 mx-auto text-muted-foreground" />
          <div>
            <h3 className="text-lg font-semibold">No Active Execution</h3>
            <p className="text-muted-foreground">Start an infinite loop to monitor execution</p>
          </div>
        </div>
      </div>
    )
  }

  const { executionState, waves, performanceMetrics } = currentSession

  return (
    <div className="space-y-6">
      {/* Execution Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Current Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              {executionState.isRunning ? (
                <>
                  <Play className="h-4 w-4 text-green-500" />
                  <Badge className="bg-green-500">Running</Badge>
                </>
              ) : (
                <>
                  <Square className="h-4 w-4 text-gray-500" />
                  <Badge variant="secondary">Stopped</Badge>
                </>
              )}
            </div>
            <div className="mt-2 text-2xl font-bold">
              Wave {executionState.currentWave}
            </div>
            <p className="text-sm text-muted-foreground">
              {executionState.totalIterations} iterations completed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Success Rate</span>
                <span>{(executionState.successRate * 100).toFixed(1)}%</span>
              </div>
              <Progress value={executionState.successRate * 100} />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Completed: {executionState.completedIterations.length}</span>
                <span>Failed: {executionState.failedIterations.length}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <Clock className="h-4 w-4 text-blue-500" />
                <span className="text-sm">Avg Time</span>
              </div>
              <div className="text-2xl font-bold">
                {executionState.averageIterationTime.toFixed(1)}s
              </div>
              <p className="text-sm text-muted-foreground">
                per iteration
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Wave Timeline */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Zap className="h-5 w-5" />
            <span>Wave Timeline</span>
          </CardTitle>
          <CardDescription>
            Real-time execution progress across waves
          </CardDescription>
        </CardHeader>
        <CardContent>
          {waves.length > 0 ? (
            <div className="space-y-4">
              {waves.map((wave, index) => (
                <div key={index} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline">Wave {wave.waveNumber}</Badge>
                      <span className="text-sm text-muted-foreground">
                        {new Date(wave.startTime).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      {wave.completedIterations > 0 ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <AlertCircle className="h-4 w-4 text-yellow-500" />
                      )}
                      <span className="text-sm">
                        {wave.completedIterations}/{wave.completedIterations + wave.failedIterations}
                      </span>
                    </div>
                  </div>
                  <Progress 
                    value={(wave.completedIterations / (wave.completedIterations + wave.failedIterations)) * 100} 
                    className="mb-2"
                  />
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Duration:</span>
                      <div className="font-medium">{wave.totalTime.toFixed(1)}s</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Success Rate:</span>
                      <div className="font-medium">{(wave.successRate * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Quality:</span>
                      <div className="font-medium">{(wave.qualityScore * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Zap className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
              <p className="text-muted-foreground">No waves executed yet</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Active Tasks */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Active Tasks</span>
          </CardTitle>
          <CardDescription>
            Currently running iteration tasks
          </CardDescription>
        </CardHeader>
        <CardContent>
          {Object.keys(executionState.activeAgents).length > 0 ? (
            <div className="space-y-3">
              {Object.entries(executionState.activeAgents).map(([agentId, task]) => (
                <div key={agentId} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                    <div>
                      <div className="font-medium">Agent {agentId}</div>
                      <div className="text-sm text-muted-foreground">{task}</div>
                    </div>
                  </div>
                  <Badge variant="outline">Running</Badge>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Activity className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
              <p className="text-muted-foreground">No active tasks</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
