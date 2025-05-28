'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import {
  ArrowRight,
  CheckCircle,
  Clock,
  AlertCircle,
  Play,
  Pause,
  RotateCcw,
  Eye,
  Zap,
  Database,
  Image,
  Mic,
  FileText
} from 'lucide-react'

interface PipelineStep {
  id: string
  name: string
  type: 'input' | 'processing' | 'output'
  status: 'pending' | 'running' | 'completed' | 'error'
  duration?: number
  progress?: number
  details?: string
}

interface Pipeline {
  id: string
  name: string
  type: 'multimodal' | 'rag' | 'streaming'
  status: 'idle' | 'running' | 'completed' | 'error'
  steps: PipelineStep[]
  startTime?: string
  endTime?: string
  totalDuration?: number
}

interface PipelineVisualizationProps {
  className?: string
}

const mockPipelines: Pipeline[] = [
  {
    id: 'multimodal-1',
    name: 'Image Text Extraction',
    type: 'multimodal',
    status: 'running',
    startTime: '2025-01-01T10:30:00Z',
    steps: [
      {
        id: 'input-1',
        name: 'Image Input',
        type: 'input',
        status: 'completed',
        duration: 0.1,
        details: 'business-card.jpg (2.3MB)'
      },
      {
        id: 'ocr-1',
        name: 'OCR Processing',
        type: 'processing',
        status: 'running',
        progress: 65,
        details: 'Extracting text from image...'
      },
      {
        id: 'nlp-1',
        name: 'Text Analysis',
        type: 'processing',
        status: 'pending',
        details: 'Analyzing extracted text'
      },
      {
        id: 'output-1',
        name: 'Structured Output',
        type: 'output',
        status: 'pending',
        details: 'Contact information JSON'
      }
    ]
  },
  {
    id: 'rag-1',
    name: 'Document Search',
    type: 'rag',
    status: 'completed',
    startTime: '2025-01-01T10:25:00Z',
    endTime: '2025-01-01T10:25:45Z',
    totalDuration: 45,
    steps: [
      {
        id: 'query-1',
        name: 'Query Processing',
        type: 'input',
        status: 'completed',
        duration: 0.2,
        details: 'machine learning best practices'
      },
      {
        id: 'search-1',
        name: 'Hybrid Search',
        type: 'processing',
        status: 'completed',
        duration: 1.8,
        details: 'Vector + keyword search'
      },
      {
        id: 'rank-1',
        name: 'Result Ranking',
        type: 'processing',
        status: 'completed',
        duration: 0.5,
        details: 'Semantic relevance scoring'
      },
      {
        id: 'generate-1',
        name: 'Response Generation',
        type: 'output',
        status: 'completed',
        duration: 2.1,
        details: 'Generated comprehensive guide'
      }
    ]
  },
  {
    id: 'streaming-1',
    name: 'Real-time Log Analysis',
    type: 'streaming',
    status: 'running',
    startTime: '2025-01-01T10:00:00Z',
    steps: [
      {
        id: 'stream-1',
        name: 'Log Stream',
        type: 'input',
        status: 'running',
        details: 'Processing 1.2k events/sec'
      },
      {
        id: 'filter-1',
        name: 'Event Filtering',
        type: 'processing',
        status: 'running',
        details: 'Filtering error events'
      },
      {
        id: 'analyze-1',
        name: 'Pattern Analysis',
        type: 'processing',
        status: 'running',
        details: 'Detecting anomalies'
      },
      {
        id: 'alert-1',
        name: 'Alert Generation',
        type: 'output',
        status: 'running',
        details: 'Real-time notifications'
      }
    ]
  }
]

const getStepIcon = (step: PipelineStep) => {
  if (step.type === 'input') {
    if (step.name.includes('Image')) return <Image className="h-4 w-4" />
    if (step.name.includes('Audio')) return <Mic className="h-4 w-4" />
    return <FileText className="h-4 w-4" />
  }
  if (step.type === 'output') return <Database className="h-4 w-4" />
  return <Zap className="h-4 w-4" />
}

const getStatusIcon = (status: PipelineStep['status']) => {
  switch (status) {
    case 'completed':
      return <CheckCircle className="h-4 w-4 text-green-500" />
    case 'running':
      return <Clock className="h-4 w-4 text-blue-500 animate-spin" />
    case 'error':
      return <AlertCircle className="h-4 w-4 text-red-500" />
    default:
      return <Clock className="h-4 w-4 text-gray-400" />
  }
}

const getStatusColor = (status: Pipeline['status']) => {
  switch (status) {
    case 'running':
      return 'bg-blue-500'
    case 'completed':
      return 'bg-green-500'
    case 'error':
      return 'bg-red-500'
    default:
      return 'bg-gray-500'
  }
}

const getPipelineTypeIcon = (type: Pipeline['type']) => {
  switch (type) {
    case 'multimodal':
      return <Image className="h-5 w-5" />
    case 'rag':
      return <Database className="h-5 w-5" />
    case 'streaming':
      return <Zap className="h-5 w-5" />
    default:
      return <FileText className="h-5 w-5" />
  }
}

export const PipelineVisualization: React.FC<PipelineVisualizationProps> = ({ className }) => {
  const [pipelines, setPipelines] = useState<Pipeline[]>(mockPipelines)
  const [selectedPipeline, setSelectedPipeline] = useState<Pipeline | null>(null)

  useEffect(() => {
    // Simulate real-time updates for running pipelines
    const interval = setInterval(() => {
      setPipelines(prevPipelines =>
        prevPipelines.map(pipeline => {
          if (pipeline.status === 'running') {
            return {
              ...pipeline,
              steps: pipeline.steps.map(step => {
                if (step.status === 'running' && step.progress !== undefined) {
                  return {
                    ...step,
                    progress: Math.min(100, step.progress + Math.random() * 5)
                  }
                }
                return step
              })
            }
          }
          return pipeline
        })
      )
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  const handlePipelineAction = (pipelineId: string, action: 'start' | 'stop' | 'restart') => {
    setPipelines(prevPipelines =>
      prevPipelines.map(pipeline =>
        pipeline.id === pipelineId
          ? {
              ...pipeline,
              status: action === 'start' ? 'running' : action === 'stop' ? 'idle' : 'running'
            }
          : pipeline
      )
    )
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Pipeline Visualization</h1>
          <p className="text-muted-foreground">
            Monitor and visualize LLM pipeline execution in real-time
          </p>
        </div>
        <Button>
          <Eye className="mr-2 h-4 w-4" />
          View All Pipelines
        </Button>
      </div>

      {/* Pipeline Cards */}
      <div className="grid gap-6">
        {pipelines.map((pipeline) => (
          <Card key={pipeline.id}
                className={`cursor-pointer hover:shadow-md transition-shadow ${
                  selectedPipeline?.id === pipeline.id ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() => setSelectedPipeline(selectedPipeline?.id === pipeline.id ? null : pipeline)}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {getPipelineTypeIcon(pipeline.type)}
                  <div>
                    <CardTitle className="text-lg">{pipeline.name}</CardTitle>
                    <CardDescription>
                      {pipeline.type.charAt(0).toUpperCase() + pipeline.type.slice(1)} Pipeline
                      {pipeline.startTime && (
                        <span className="ml-2">
                          Started: {new Date(pipeline.startTime).toLocaleTimeString()}
                        </span>
                      )}
                    </CardDescription>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge variant={pipeline.status === 'running' ? 'default' : 'secondary'}>
                    <div className={`w-2 h-2 rounded-full mr-2 ${getStatusColor(pipeline.status)}`} />
                    {pipeline.status}
                  </Badge>
                  <div className="flex space-x-1">
                    <Button size="sm" variant="outline"
                            onClick={(e) => { e.stopPropagation(); handlePipelineAction(pipeline.id, 'start') }}>
                      <Play className="h-3 w-3" />
                    </Button>
                    <Button size="sm" variant="outline"
                            onClick={(e) => { e.stopPropagation(); handlePipelineAction(pipeline.id, 'stop') }}>
                      <Pause className="h-3 w-3" />
                    </Button>
                    <Button size="sm" variant="outline"
                            onClick={(e) => { e.stopPropagation(); handlePipelineAction(pipeline.id, 'restart') }}>
                      <RotateCcw className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {/* Pipeline Steps Visualization */}
              <div className="flex items-center space-x-2 overflow-x-auto pb-2">
                {pipeline.steps.map((step, index) => (
                  <React.Fragment key={step.id}>
                    <div className="flex flex-col items-center min-w-[120px] p-3 rounded-lg border bg-card">
                      <div className="flex items-center space-x-2 mb-2">
                        {getStepIcon(step)}
                        {getStatusIcon(step.status)}
                      </div>
                      <div className="text-xs font-medium text-center mb-1">{step.name}</div>
                      {step.progress !== undefined && step.status === 'running' && (
                        <div className="w-full mb-1">
                          <Progress value={step.progress} className="h-1" />
                          <div className="text-xs text-center text-muted-foreground">
                            {Math.round(step.progress)}%
                          </div>
                        </div>
                      )}
                      {step.duration && (
                        <div className="text-xs text-muted-foreground">
                          {step.duration}s
                        </div>
                      )}
                      <div className="text-xs text-muted-foreground text-center mt-1">
                        {step.details}
                      </div>
                    </div>
                    {index < pipeline.steps.length - 1 && (
                      <ArrowRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                    )}
                  </React.Fragment>
                ))}
              </div>

              {/* Pipeline Summary */}
              <div className="mt-4 pt-4 border-t">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Total Steps</p>
                    <p className="font-medium">{pipeline.steps.length}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Completed</p>
                    <p className="font-medium">
                      {pipeline.steps.filter(s => s.status === 'completed').length}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Running</p>
                    <p className="font-medium">
                      {pipeline.steps.filter(s => s.status === 'running').length}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Duration</p>
                    <p className="font-medium">
                      {pipeline.totalDuration ? `${pipeline.totalDuration}s` : 'In progress...'}
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Selected Pipeline Details */}
      {selectedPipeline && (
        <Card className="mt-6">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-xl">Pipeline Details: {selectedPipeline.name}</CardTitle>
                <CardDescription>
                  Detailed view of {selectedPipeline.type} pipeline execution
                </CardDescription>
              </div>
              <Button variant="outline" onClick={() => setSelectedPipeline(null)}>
                Close Details
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Pipeline Metadata */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Pipeline ID</p>
                  <p className="text-sm">{selectedPipeline.id}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Type</p>
                  <p className="text-sm capitalize">{selectedPipeline.type}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Status</p>
                  <Badge variant={selectedPipeline.status === 'running' ? 'default' : 'secondary'}>
                    {selectedPipeline.status}
                  </Badge>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Duration</p>
                  <p className="text-sm">
                    {selectedPipeline.totalDuration ? `${selectedPipeline.totalDuration}s` : 'In progress...'}
                  </p>
                </div>
              </div>

              {/* Step Details */}
              <div>
                <h4 className="text-lg font-medium mb-4">Step Details</h4>
                <div className="space-y-3">
                  {selectedPipeline.steps.map((step, index) => (
                    <div key={step.id} className="flex items-center space-x-4 p-3 border rounded-lg">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-medium w-8">{index + 1}.</span>
                        {getStepIcon(step)}
                        {getStatusIcon(step.status)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <p className="font-medium">{step.name}</p>
                          <div className="flex items-center space-x-2">
                            {step.duration && (
                              <span className="text-sm text-muted-foreground">{step.duration}s</span>
                            )}
                            {step.progress !== undefined && step.status === 'running' && (
                              <div className="flex items-center space-x-2">
                                <Progress value={step.progress} className="w-20 h-2" />
                                <span className="text-xs">{Math.round(step.progress)}%</span>
                              </div>
                            )}
                          </div>
                        </div>
                        <p className="text-sm text-muted-foreground">{step.details}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
