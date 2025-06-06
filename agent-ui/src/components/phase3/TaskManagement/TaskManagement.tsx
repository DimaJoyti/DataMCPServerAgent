'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import {
  Clock,
  CheckCircle,
  AlertCircle,
  Play,
  Pause,
  X,
  Plus,
  Filter,
  Timer
} from 'lucide-react'

interface Task {
  id: string
  title: string
  description: string
  type: 'multimodal' | 'rag' | 'streaming' | 'standard'
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  priority: 'low' | 'medium' | 'high' | 'urgent'
  assignedAgent?: string
  createdAt: string
  startedAt?: string
  completedAt?: string
  progress?: number
  estimatedDuration?: number
  actualDuration?: number
  result?: string
  error?: string
}

// TaskQueue interface for future use
// interface TaskQueue {
//   name: string
//   tasks: Task[]
//   maxConcurrent: number
//   currentRunning: number
// }

interface TaskManagementProps {
  className?: string
}

const mockTasks: Task[] = [
  {
    id: 'task-1',
    title: 'Extract text from business cards',
    description: 'Process 50 business card images and extract contact information',
    type: 'multimodal',
    status: 'running',
    priority: 'high',
    assignedAgent: 'multimodal-agent-1',
    createdAt: '2025-01-01T10:00:00Z',
    startedAt: '2025-01-01T10:05:00Z',
    progress: 68,
    estimatedDuration: 300,
  },
  {
    id: 'task-2',
    title: 'Search technical documentation',
    description: 'Find relevant documentation for API integration best practices',
    type: 'rag',
    status: 'completed',
    priority: 'medium',
    assignedAgent: 'rag-agent-1',
    createdAt: '2025-01-01T09:30:00Z',
    startedAt: '2025-01-01T09:32:00Z',
    completedAt: '2025-01-01T09:45:00Z',
    actualDuration: 780,
    result: 'Found 15 relevant documents with 95% confidence'
  },
  {
    id: 'task-3',
    title: 'Monitor system logs',
    description: 'Real-time analysis of application logs for error patterns',
    type: 'streaming',
    status: 'running',
    priority: 'urgent',
    assignedAgent: 'streaming-agent-1',
    createdAt: '2025-01-01T08:00:00Z',
    startedAt: '2025-01-01T08:01:00Z',
    progress: 100, // Continuous task
  },
  {
    id: 'task-4',
    title: 'Generate monthly report',
    description: 'Compile and analyze data for monthly performance report',
    type: 'standard',
    status: 'queued',
    priority: 'low',
    createdAt: '2025-01-01T10:30:00Z',
    estimatedDuration: 1800,
  },
  {
    id: 'task-5',
    title: 'Process audio transcriptions',
    description: 'Transcribe and analyze customer service calls',
    type: 'multimodal',
    status: 'failed',
    priority: 'medium',
    assignedAgent: 'multimodal-agent-2',
    createdAt: '2025-01-01T09:00:00Z',
    startedAt: '2025-01-01T09:15:00Z',
    error: 'Audio format not supported'
  }
]

const getStatusIcon = (status: Task['status']) => {
  switch (status) {
    case 'completed':
      return <CheckCircle className="h-4 w-4 text-green-500" />
    case 'running':
      return <Clock className="h-4 w-4 text-blue-500 animate-spin" />
    case 'failed':
      return <AlertCircle className="h-4 w-4 text-red-500" />
    case 'cancelled':
      return <X className="h-4 w-4 text-gray-500" />
    default:
      return <Clock className="h-4 w-4 text-gray-400" />
  }
}

const getStatusColor = (status: Task['status']) => {
  switch (status) {
    case 'running':
      return 'bg-blue-500'
    case 'completed':
      return 'bg-green-500'
    case 'failed':
      return 'bg-red-500'
    case 'cancelled':
      return 'bg-gray-500'
    default:
      return 'bg-yellow-500'
  }
}

const getPriorityColor = (priority: Task['priority']) => {
  switch (priority) {
    case 'urgent':
      return 'bg-red-500'
    case 'high':
      return 'bg-orange-500'
    case 'medium':
      return 'bg-yellow-500'
    default:
      return 'bg-green-500'
  }
}

const formatDuration = (seconds: number) => {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`
  } else {
    return `${secs}s`
  }
}

export const TaskManagement: React.FC<TaskManagementProps> = ({ className }) => {
  const [tasks, setTasks] = useState<Task[]>(mockTasks)
  const [selectedTask, setSelectedTask] = useState<Task | null>(null)
  const [filterStatus, setFilterStatus] = useState<string>('all')
  // const [filterType, setFilterType] = useState<string>('all') // For future use

  useEffect(() => {
    // Simulate real-time updates for running tasks
    const interval = setInterval(() => {
      setTasks(prevTasks =>
        prevTasks.map(task => {
          if (task.status === 'running' && task.progress !== undefined && task.progress < 100) {
            return {
              ...task,
              progress: Math.min(100, task.progress + Math.random() * 3)
            }
          }
          return task
        })
      )
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const handleTaskAction = (taskId: string, action: 'start' | 'pause' | 'cancel' | 'retry') => {
    setTasks(prevTasks =>
      prevTasks.map(task =>
        task.id === taskId
          ? {
              ...task,
              status: action === 'start' ? 'running' :
                     action === 'pause' ? 'queued' :
                     action === 'cancel' ? 'cancelled' :
                     action === 'retry' ? 'queued' : task.status,
              startedAt: action === 'start' ? new Date().toISOString() : task.startedAt,
              progress: action === 'retry' ? 0 : task.progress,
              error: action === 'retry' ? undefined : task.error
            }
          : task
      )
    )
  }

  const filteredTasks = tasks.filter(task => {
    const statusMatch = filterStatus === 'all' || task.status === filterStatus
    // const typeMatch = filterType === 'all' || task.type === filterType // For future use
    return statusMatch // && typeMatch
  })

  const taskStats = {
    total: tasks.length,
    queued: tasks.filter(t => t.status === 'queued').length,
    running: tasks.filter(t => t.status === 'running').length,
    completed: tasks.filter(t => t.status === 'completed').length,
    failed: tasks.filter(t => t.status === 'failed').length,
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Task Management</h1>
          <p className="text-muted-foreground">
            Monitor and manage task execution across all agents
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline">
            <Filter className="mr-2 h-4 w-4" />
            Filters
          </Button>
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            New Task
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Tasks</CardTitle>
            <Timer className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{taskStats.total}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Queued</CardTitle>
            <Clock className="h-4 w-4 text-yellow-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{taskStats.queued}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Running</CardTitle>
            <Play className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{taskStats.running}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Completed</CardTitle>
            <CheckCircle className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{taskStats.completed}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Failed</CardTitle>
            <AlertCircle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{taskStats.failed}</div>
          </CardContent>
        </Card>
      </div>

      {/* Task List */}
      <Tabs defaultValue="all" className="space-y-4">
        <TabsList>
          <TabsTrigger value="all" onClick={() => setFilterStatus('all')}>All</TabsTrigger>
          <TabsTrigger value="queued" onClick={() => setFilterStatus('queued')}>Queued</TabsTrigger>
          <TabsTrigger value="running" onClick={() => setFilterStatus('running')}>Running</TabsTrigger>
          <TabsTrigger value="completed" onClick={() => setFilterStatus('completed')}>Completed</TabsTrigger>
          <TabsTrigger value="failed" onClick={() => setFilterStatus('failed')}>Failed</TabsTrigger>
        </TabsList>

        <TabsContent value={filterStatus} className="space-y-4">
          <div className="grid gap-4">
            {filteredTasks.map((task) => (
              <Card key={task.id}
                    className={`cursor-pointer hover:shadow-md transition-shadow ${
                      selectedTask?.id === task.id ? 'ring-2 ring-primary' : ''
                    }`}
                    onClick={() => setSelectedTask(selectedTask?.id === task.id ? null : task)}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(task.status)}
                      <div>
                        <CardTitle className="text-lg">{task.title}</CardTitle>
                        <CardDescription>{task.description}</CardDescription>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline">
                        <div className={`w-2 h-2 rounded-full mr-2 ${getPriorityColor(task.priority)}`} />
                        {task.priority}
                      </Badge>
                      <Badge variant="secondary">{task.type}</Badge>
                      <Badge variant={task.status === 'running' ? 'default' : 'outline'}>
                        <div className={`w-2 h-2 rounded-full mr-2 ${getStatusColor(task.status)}`} />
                        {task.status}
                      </Badge>
                      <div className="flex space-x-1">
                        {task.status === 'queued' && (
                          <Button size="sm" variant="outline"
                                  onClick={(e) => { e.stopPropagation(); handleTaskAction(task.id, 'start') }}>
                            <Play className="h-3 w-3" />
                          </Button>
                        )}
                        {task.status === 'running' && (
                          <Button size="sm" variant="outline"
                                  onClick={(e) => { e.stopPropagation(); handleTaskAction(task.id, 'pause') }}>
                            <Pause className="h-3 w-3" />
                          </Button>
                        )}
                        {task.status === 'failed' && (
                          <Button size="sm" variant="outline"
                                  onClick={(e) => { e.stopPropagation(); handleTaskAction(task.id, 'retry') }}>
                            <Play className="h-3 w-3" />
                          </Button>
                        )}
                        <Button size="sm" variant="outline"
                                onClick={(e) => { e.stopPropagation(); handleTaskAction(task.id, 'cancel') }}>
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Agent</p>
                      <p className="font-medium">{task.assignedAgent || 'Unassigned'}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Created</p>
                      <p className="font-medium">
                        {new Date(task.createdAt).toLocaleTimeString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Duration</p>
                      <p className="font-medium">
                        {task.actualDuration ? formatDuration(task.actualDuration) :
                         task.estimatedDuration ? `~${formatDuration(task.estimatedDuration)}` : 'Unknown'}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Progress</p>
                      {task.progress !== undefined ? (
                        <div className="flex items-center space-x-2">
                          <Progress value={task.progress} className="flex-1" />
                          <span className="text-xs">{Math.round(task.progress)}%</span>
                        </div>
                      ) : (
                        <p className="font-medium">-</p>
                      )}
                    </div>
                  </div>

                  {task.error && (
                    <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                      <strong>Error:</strong> {task.error}
                    </div>
                  )}

                  {task.result && (
                    <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded text-sm text-green-700">
                      <strong>Result:</strong> {task.result}
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>

      {/* Selected Task Details */}
      {selectedTask && (
        <Card className="mt-6">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-xl">Task Details: {selectedTask.title}</CardTitle>
                <CardDescription>
                  Detailed view of {selectedTask.type} task execution
                </CardDescription>
              </div>
              <Button variant="outline" onClick={() => setSelectedTask(null)}>
                Close Details
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Task Metadata */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Task ID</p>
                  <p className="text-sm">{selectedTask.id}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Type</p>
                  <p className="text-sm capitalize">{selectedTask.type}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Priority</p>
                  <Badge variant="outline">
                    <div className={`w-2 h-2 rounded-full mr-2 ${getPriorityColor(selectedTask.priority)}`} />
                    {selectedTask.priority}
                  </Badge>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Status</p>
                  <Badge variant={selectedTask.status === 'running' ? 'default' : 'outline'}>
                    <div className={`w-2 h-2 rounded-full mr-2 ${getStatusColor(selectedTask.status)}`} />
                    {selectedTask.status}
                  </Badge>
                </div>
              </div>

              {/* Task Description */}
              <div>
                <h4 className="text-lg font-medium mb-2">Description</h4>
                <p className="text-sm text-muted-foreground">{selectedTask.description}</p>
              </div>

              {/* Task Timeline */}
              <div>
                <h4 className="text-lg font-medium mb-4">Timeline</h4>
                <div className="space-y-3">
                  <div className="flex items-center space-x-4">
                    <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                    <div>
                      <p className="text-sm font-medium">Created</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(selectedTask.createdAt).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  {selectedTask.startedAt && (
                    <div className="flex items-center space-x-4">
                      <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                      <div>
                        <p className="text-sm font-medium">Started</p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(selectedTask.startedAt).toLocaleString()}
                        </p>
                      </div>
                    </div>
                  )}
                  {selectedTask.completedAt && (
                    <div className="flex items-center space-x-4">
                      <div className="w-2 h-2 rounded-full bg-green-500"></div>
                      <div>
                        <p className="text-sm font-medium">Completed</p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(selectedTask.completedAt).toLocaleString()}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Task Progress */}
              {selectedTask.progress !== undefined && (
                <div>
                  <h4 className="text-lg font-medium mb-2">Progress</h4>
                  <div className="flex items-center space-x-4">
                    <Progress value={selectedTask.progress} className="flex-1" />
                    <span className="text-sm font-medium">{Math.round(selectedTask.progress)}%</span>
                  </div>
                </div>
              )}

              {/* Task Result */}
              {selectedTask.result && (
                <div>
                  <h4 className="text-lg font-medium mb-2">Result</h4>
                  <div className="p-3 bg-green-50 border border-green-200 rounded text-sm">
                    {selectedTask.result}
                  </div>
                </div>
              )}

              {/* Task Error */}
              {selectedTask.error && (
                <div>
                  <h4 className="text-lg font-medium mb-2">Error</h4>
                  <div className="p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                    {selectedTask.error}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
