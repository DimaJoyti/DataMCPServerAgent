'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import {
  Activity,
  Brain,
  Cpu,
  Database,
  Play,
  Pause,
  Settings,
  Zap,
  Search,
  Image
} from 'lucide-react'

interface Agent {
  id: string
  name: string
  type: 'multimodal' | 'rag' | 'streaming' | 'standard'
  status: 'active' | 'idle' | 'busy' | 'error'
  capabilities: string[]
  performance: {
    tasksCompleted: number
    averageResponseTime: number
    successRate: number
    currentLoad: number
  }
  lastActivity: string
}

interface AgentDashboardProps {
  className?: string
}

const mockAgents: Agent[] = [
  {
    id: 'multimodal-1',
    name: 'Multimodal Agent',
    type: 'multimodal',
    status: 'active',
    capabilities: ['text_image_processing', 'text_audio_processing', 'ocr', 'speech_recognition'],
    performance: {
      tasksCompleted: 156,
      averageResponseTime: 1.2,
      successRate: 98.5,
      currentLoad: 45
    },
    lastActivity: '2 minutes ago'
  },
  {
    id: 'rag-1',
    name: 'RAG Agent',
    type: 'rag',
    status: 'busy',
    capabilities: ['hybrid_search', 'document_retrieval', 'context_generation'],
    performance: {
      tasksCompleted: 89,
      averageResponseTime: 0.8,
      successRate: 99.1,
      currentLoad: 78
    },
    lastActivity: 'Just now'
  },
  {
    id: 'streaming-1',
    name: 'Streaming Agent',
    type: 'streaming',
    status: 'active',
    capabilities: ['real_time_processing', 'incremental_updates', 'live_monitoring'],
    performance: {
      tasksCompleted: 234,
      averageResponseTime: 0.3,
      successRate: 97.8,
      currentLoad: 23
    },
    lastActivity: '5 seconds ago'
  }
]

const getAgentIcon = (type: Agent['type']) => {
  switch (type) {
    case 'multimodal':
      return <Image className="h-5 w-5" />
    case 'rag':
      return <Search className="h-5 w-5" />
    case 'streaming':
      return <Zap className="h-5 w-5" />
    default:
      return <Brain className="h-5 w-5" />
  }
}

const getStatusColor = (status: Agent['status']) => {
  switch (status) {
    case 'active':
      return 'bg-green-500'
    case 'busy':
      return 'bg-yellow-500'
    case 'idle':
      return 'bg-gray-500'
    case 'error':
      return 'bg-red-500'
    default:
      return 'bg-gray-500'
  }
}

const getStatusBadgeVariant = (status: Agent['status']) => {
  switch (status) {
    case 'active':
      return 'default'
    case 'busy':
      return 'secondary'
    case 'idle':
      return 'outline'
    case 'error':
      return 'destructive'
    default:
      return 'outline'
  }
}

export const AgentDashboard: React.FC<AgentDashboardProps> = ({ className }) => {
  const [agents, setAgents] = useState<Agent[]>(mockAgents)
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [isHydrated, setIsHydrated] = useState(false)

  useEffect(() => {
    setIsHydrated(true)

    // Simulate real-time updates only after hydration
    const interval = setInterval(() => {
      setAgents(prevAgents =>
        prevAgents.map(agent => ({
          ...agent,
          performance: {
            ...agent.performance,
            currentLoad: Math.max(0, Math.min(100, agent.performance.currentLoad + (Math.random() - 0.5) * 10))
          }
        }))
      )
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const handleAgentAction = (agentId: string, action: 'start' | 'stop' | 'restart') => {
    setAgents(prevAgents =>
      prevAgents.map(agent =>
        agent.id === agentId
          ? {
              ...agent,
              status: action === 'start' ? 'active' : action === 'stop' ? 'idle' : 'busy'
            }
          : agent
      )
    )
  }

  if (!isHydrated) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg animate-pulse">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <p className="text-slate-600 dark:text-slate-400">Loading Agent Dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="space-y-2">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
              <Brain className="h-7 w-7 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-300 bg-clip-text text-transparent">
                Agent Dashboard
              </h1>
              <p className="text-slate-600 dark:text-slate-400 text-lg">
                Manage and monitor your integrated semantic agents
              </p>
            </div>
          </div>
        </div>
        <Button
          size="lg"
          className="bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 shadow-lg shadow-indigo-500/25 text-white border-0"
        >
          <Settings className="mr-2 h-5 w-5" />
          Configure Agents
        </Button>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card className="border-0 shadow-lg bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-900/20 dark:to-indigo-900/20">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-sm font-semibold text-blue-700 dark:text-blue-300">Total Agents</CardTitle>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
              <Brain className="h-5 w-5 text-white" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-blue-900 dark:text-blue-100">{agents.length}</div>
            <p className="text-sm text-blue-600 dark:text-blue-400 mt-1">
              {agents.filter(a => a.status === 'active').length} active
            </p>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-lg bg-gradient-to-br from-emerald-50 to-teal-100 dark:from-emerald-900/20 dark:to-teal-900/20">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-sm font-semibold text-emerald-700 dark:text-emerald-300">Tasks Completed</CardTitle>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg">
              <Activity className="h-5 w-5 text-white" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-emerald-900 dark:text-emerald-100">
              {agents.reduce((sum, agent) => sum + agent.performance.tasksCompleted, 0)}
            </div>
            <p className="text-sm text-emerald-600 dark:text-emerald-400 mt-1">
              +12% from last hour
            </p>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-lg bg-gradient-to-br from-purple-50 to-violet-100 dark:from-purple-900/20 dark:to-violet-900/20">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-sm font-semibold text-purple-700 dark:text-purple-300">Avg Response Time</CardTitle>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-violet-600 flex items-center justify-center shadow-lg">
              <Cpu className="h-5 w-5 text-white" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-purple-900 dark:text-purple-100">
              {(agents.reduce((sum, agent) => sum + agent.performance.averageResponseTime, 0) / agents.length).toFixed(1)}s
            </div>
            <p className="text-sm text-purple-600 dark:text-purple-400 mt-1">
              -0.2s from last hour
            </p>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-lg bg-gradient-to-br from-amber-50 to-orange-100 dark:from-amber-900/20 dark:to-orange-900/20">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-sm font-semibold text-amber-700 dark:text-amber-300">Success Rate</CardTitle>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center shadow-lg">
              <Database className="h-5 w-5 text-white" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-amber-900 dark:text-amber-100">
              {(agents.reduce((sum, agent) => sum + agent.performance.successRate, 0) / agents.length).toFixed(1)}%
            </div>
            <p className="text-sm text-amber-600 dark:text-amber-400 mt-1">
              +0.3% from last hour
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Agents List */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="multimodal">Multimodal</TabsTrigger>
          <TabsTrigger value="rag">RAG</TabsTrigger>
          <TabsTrigger value="streaming">Streaming</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid gap-6">
            {agents.map((agent) => (
              <Card key={agent.id}
                    className={`cursor-pointer hover:shadow-xl transition-all duration-300 border-0 shadow-lg bg-gradient-to-r from-white to-slate-50 dark:from-slate-800 dark:to-slate-900 ${
                      selectedAgent?.id === agent.id ? 'ring-2 ring-blue-500 shadow-blue-500/25' : ''
                    }`}
                    onClick={() => setSelectedAgent(selectedAgent?.id === agent.id ? null : agent)}>
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
                        {getAgentIcon(agent.type)}
                      </div>
                      <div>
                        <CardTitle className="text-xl font-bold text-slate-900 dark:text-white">{agent.name}</CardTitle>
                        <CardDescription className="text-slate-600 dark:text-slate-400 text-base">
                          {agent.capabilities.slice(0, 2).join(', ')}
                          {agent.capabilities.length > 2 && ` +${agent.capabilities.length - 2} more`}
                        </CardDescription>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <Badge
                        variant={getStatusBadgeVariant(agent.status)}
                        className={`px-3 py-1 text-sm font-medium ${
                          agent.status === 'active' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' :
                          agent.status === 'busy' ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400' :
                          'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-400'
                        }`}
                      >
                        <div className={`w-2 h-2 rounded-full mr-2 ${getStatusColor(agent.status)} shadow-lg`} />
                        {agent.status}
                      </Badge>
                      <div className="flex space-x-2">
                        <Button size="sm" variant="outline"
                                className="hover:bg-emerald-50 hover:border-emerald-300 hover:text-emerald-700 transition-colors"
                                onClick={(e) => { e.stopPropagation(); handleAgentAction(agent.id, 'start') }}>
                          <Play className="h-4 w-4" />
                        </Button>
                        <Button size="sm" variant="outline"
                                className="hover:bg-red-50 hover:border-red-300 hover:text-red-700 transition-colors"
                                onClick={(e) => { e.stopPropagation(); handleAgentAction(agent.id, 'stop') }}>
                          <Pause className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    <div className="text-center p-3 rounded-xl bg-blue-50 dark:bg-blue-900/20">
                      <p className="text-xs font-medium text-blue-600 dark:text-blue-400 uppercase tracking-wider">Tasks</p>
                      <p className="text-2xl font-bold text-blue-900 dark:text-blue-100 mt-1">{agent.performance.tasksCompleted}</p>
                    </div>
                    <div className="text-center p-3 rounded-xl bg-purple-50 dark:bg-purple-900/20">
                      <p className="text-xs font-medium text-purple-600 dark:text-purple-400 uppercase tracking-wider">Response Time</p>
                      <p className="text-2xl font-bold text-purple-900 dark:text-purple-100 mt-1">{agent.performance.averageResponseTime}s</p>
                    </div>
                    <div className="text-center p-3 rounded-xl bg-emerald-50 dark:bg-emerald-900/20">
                      <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">Success Rate</p>
                      <p className="text-2xl font-bold text-emerald-900 dark:text-emerald-100 mt-1">{agent.performance.successRate}%</p>
                    </div>
                    <div className="text-center p-3 rounded-xl bg-amber-50 dark:bg-amber-900/20">
                      <p className="text-xs font-medium text-amber-600 dark:text-amber-400 uppercase tracking-wider">Load</p>
                      <div className="mt-2">
                        <Progress
                          value={agent.performance.currentLoad}
                          className="h-2 bg-amber-200 dark:bg-amber-800"
                        />
                        <span className="text-lg font-bold text-amber-900 dark:text-amber-100 mt-1 block">
                          {Math.round(agent.performance.currentLoad)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700">
                    <div className="flex items-center justify-between">
                      <p className="text-sm text-slate-600 dark:text-slate-400">
                        Last activity: <span className="font-medium">{agent.lastActivity}</span>
                      </p>
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                        <span className="text-xs text-slate-500">Live</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Type-specific tabs would filter agents by type */}
        <TabsContent value="multimodal">
          <div className="text-center py-8">
            <Image className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">Multimodal Agents</h3>
            <p className="text-muted-foreground">
              Agents specialized in processing text, images, and audio content
            </p>
          </div>
        </TabsContent>

        <TabsContent value="rag">
          <div className="text-center py-8">
            <Search className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">RAG Agents</h3>
            <p className="text-muted-foreground">
              Agents specialized in retrieval-augmented generation and knowledge search
            </p>
          </div>
        </TabsContent>

        <TabsContent value="streaming">
          <div className="text-center py-8">
            <Zap className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">Streaming Agents</h3>
            <p className="text-muted-foreground">
              Agents specialized in real-time processing and live data streams
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
