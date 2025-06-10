'use client'

import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Clock,
  Target,
  Zap,
  Award
} from 'lucide-react'
import { useInfiniteLoopStore } from '@/store'

export const AnalyticsDashboard: React.FC = () => {
  const { currentSession } = useInfiniteLoopStore()

  if (!currentSession) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <BarChart3 className="h-12 w-12 mx-auto text-muted-foreground" />
          <div>
            <h3 className="text-lg font-semibold">No Analytics Data</h3>
            <p className="text-muted-foreground">Start an infinite loop to view analytics</p>
          </div>
        </div>
      </div>
    )
  }

  const { performanceMetrics, qualityMetrics, waves } = currentSession

  // Calculate trends
  const getTrend = (current: number, previous: number) => {
    if (previous === 0) return 0
    return ((current - previous) / previous) * 100
  }

  const recentWaves = waves.slice(-5) // Last 5 waves
  const avgRecentQuality = recentWaves.length > 0 
    ? recentWaves.reduce((sum, wave) => sum + wave.qualityScore, 0) / recentWaves.length
    : 0

  const avgRecentSuccess = recentWaves.length > 0 
    ? recentWaves.reduce((sum, wave) => sum + wave.successRate, 0) / recentWaves.length
    : 0

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center space-x-2">
              <Target className="h-4 w-4" />
              <span>Success Rate</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(performanceMetrics.successRate * 100).toFixed(1)}%
            </div>
            <div className="flex items-center space-x-1 text-sm">
              {avgRecentSuccess > performanceMetrics.successRate ? (
                <TrendingUp className="h-3 w-3 text-green-500" />
              ) : (
                <TrendingDown className="h-3 w-3 text-red-500" />
              )}
              <span className="text-muted-foreground">
                vs recent avg
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center space-x-2">
              <Award className="h-4 w-4" />
              <span>Quality Score</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(qualityMetrics.overallScore * 100).toFixed(1)}%
            </div>
            <div className="flex items-center space-x-1 text-sm">
              {avgRecentQuality > qualityMetrics.overallScore ? (
                <TrendingUp className="h-3 w-3 text-green-500" />
              ) : (
                <TrendingDown className="h-3 w-3 text-red-500" />
              )}
              <span className="text-muted-foreground">
                trending {avgRecentQuality > qualityMetrics.overallScore ? 'up' : 'down'}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center space-x-2">
              <Clock className="h-4 w-4" />
              <span>Avg Time</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {performanceMetrics.averageIterationTime.toFixed(1)}s
            </div>
            <p className="text-sm text-muted-foreground">
              per iteration
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center space-x-2">
              <Zap className="h-4 w-4" />
              <span>Throughput</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {performanceMetrics.throughput.toFixed(1)}
            </div>
            <p className="text-sm text-muted-foreground">
              iterations/min
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Quality Breakdown */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Award className="h-5 w-5" />
            <span>Quality Metrics Breakdown</span>
          </CardTitle>
          <CardDescription>
            Detailed quality assessment across different dimensions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Uniqueness</span>
                <span>{(qualityMetrics.uniquenessScore * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full" 
                  style={{ width: `${qualityMetrics.uniquenessScore * 100}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Validation</span>
                <span>{(qualityMetrics.validationScore * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-green-500 h-2 rounded-full" 
                  style={{ width: `${qualityMetrics.validationScore * 100}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Consistency</span>
                <span>{(qualityMetrics.consistencyScore * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-purple-500 h-2 rounded-full" 
                  style={{ width: `${qualityMetrics.consistencyScore * 100}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Innovation</span>
                <span>{(qualityMetrics.innovationScore * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-orange-500 h-2 rounded-full" 
                  style={{ width: `${qualityMetrics.innovationScore * 100}%` }}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Wave Performance Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Wave Performance Trends</span>
          </CardTitle>
          <CardDescription>
            Performance metrics across execution waves
          </CardDescription>
        </CardHeader>
        <CardContent>
          {waves.length > 0 ? (
            <div className="space-y-4">
              {/* Simple chart representation */}
              <div className="grid grid-cols-1 gap-4">
                {waves.slice(-10).map((wave, index) => (
                  <div key={index} className="flex items-center space-x-4">
                    <div className="w-16 text-sm font-medium">
                      Wave {wave.waveNumber}
                    </div>
                    <div className="flex-1 space-y-1">
                      <div className="flex justify-between text-xs">
                        <span>Success: {(wave.successRate * 100).toFixed(1)}%</span>
                        <span>Quality: {(wave.qualityScore * 100).toFixed(1)}%</span>
                        <span>Time: {wave.totalTime.toFixed(1)}s</span>
                      </div>
                      <div className="flex space-x-1">
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-green-500 h-2 rounded-full" 
                            style={{ width: `${wave.successRate * 100}%` }}
                          />
                        </div>
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-500 h-2 rounded-full" 
                            style={{ width: `${wave.qualityScore * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {wave.completedIterations} completed
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <Activity className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
              <p className="text-muted-foreground">No wave data available</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* System Performance */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5" />
            <span>System Performance</span>
          </CardTitle>
          <CardDescription>
            Resource utilization and system metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Memory Usage</span>
                <span>{(performanceMetrics.memoryUsage * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    performanceMetrics.memoryUsage > 0.8 ? 'bg-red-500' : 
                    performanceMetrics.memoryUsage > 0.6 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${performanceMetrics.memoryUsage * 100}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Context Usage</span>
                <span>{(performanceMetrics.contextUsage * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    performanceMetrics.contextUsage > 0.8 ? 'bg-red-500' : 
                    performanceMetrics.contextUsage > 0.6 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${performanceMetrics.contextUsage * 100}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Execution Time</span>
                <span>{(performanceMetrics.executionTimeSeconds / 60).toFixed(1)}m</span>
              </div>
              <div className="text-xs text-muted-foreground">
                Total runtime
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
