'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { TextArea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  Play,
  Square,
  RotateCcw,
  Upload,
  Settings,
  Zap,
  Image,
  FileText,
  Search,
  Activity,
  Clock
} from 'lucide-react'

interface PlaygroundRequest {
  id: string
  agentType: 'multimodal' | 'rag' | 'streaming' | 'standard'
  input: string
  files?: File[]
  parameters?: Record<string, unknown>
  timestamp: string
  status: 'pending' | 'running' | 'completed' | 'error'
  response?: string
  duration?: number
  error?: string
}

interface AgentPlaygroundProps {
  className?: string
}

const agentTypes = [
  {
    value: 'multimodal',
    label: 'Multimodal Agent',
    description: 'Process text, images, and audio',
    icon: <Image className="h-4 w-4" />,
    capabilities: ['Text+Image', 'Text+Audio', 'OCR', 'Speech Recognition']
  },
  {
    value: 'rag',
    label: 'RAG Agent',
    description: 'Retrieval-augmented generation',
    icon: <Search className="h-4 w-4" />,
    capabilities: ['Document Search', 'Context Generation', 'Hybrid Search']
  },
  {
    value: 'streaming',
    label: 'Streaming Agent',
    description: 'Real-time processing',
    icon: <Zap className="h-4 w-4" />,
    capabilities: ['Live Processing', 'Event Streams', 'Incremental Updates']
  },
  {
    value: 'standard',
    label: 'Standard Agent',
    description: 'General purpose processing',
    icon: <FileText className="h-4 w-4" />,
    capabilities: ['Text Processing', 'Analysis', 'Generation']
  }
]

const examplePrompts = {
  multimodal: [
    'Analyze this image and extract any text content',
    'Transcribe this audio file and identify the speaker sentiment',
    'Compare the text content with the audio narration'
  ],
  rag: [
    'Find documentation about API authentication best practices',
    'Search for examples of machine learning model deployment',
    'What are the latest trends in cloud computing?'
  ],
  streaming: [
    'Monitor incoming log events for error patterns',
    'Process real-time user activity data',
    'Analyze streaming sensor data for anomalies'
  ],
  standard: [
    'Summarize this document',
    'Generate a technical report',
    'Analyze the sentiment of this text'
  ]
}

export const AgentPlayground: React.FC<AgentPlaygroundProps> = ({ className }) => {
  const [selectedAgent, setSelectedAgent] = useState<string>('multimodal')
  const [input, setInput] = useState<string>('')
  const [requests, setRequests] = useState<PlaygroundRequest[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])

  const handleSubmit = async () => {
    if (!input.trim()) return

    const newRequest: PlaygroundRequest = {
      id: `req-${Date.now()}`,
      agentType: selectedAgent as 'multimodal' | 'rag' | 'streaming' | 'standard',
      input: input.trim(),
      files: uploadedFiles.length > 0 ? uploadedFiles : undefined,
      timestamp: new Date().toISOString(),
      status: 'pending'
    }

    setRequests(prev => [newRequest, ...prev])
    setIsRunning(true)

    // Simulate processing
    setTimeout(() => {
      setRequests(prev => prev.map(req =>
        req.id === newRequest.id
          ? { ...req, status: 'running' }
          : req
      ))
    }, 500)

    setTimeout(() => {
      const mockResponse = generateMockResponse(selectedAgent, input)
      setRequests(prev => prev.map(req =>
        req.id === newRequest.id
          ? {
              ...req,
              status: 'completed',
              response: mockResponse,
              duration: Math.random() * 3 + 1
            }
          : req
      ))
      setIsRunning(false)
    }, 2000 + Math.random() * 3000)
  }

  const generateMockResponse = (agentType: string, input: string): string => {
    switch (agentType) {
      case 'multimodal':
        return `Processed multimodal content:\n\n✅ Text analysis completed\n✅ Image/audio processing completed\n\nExtracted information:\n- Content type: Mixed media\n- Processing time: 1.2s\n- Confidence: 94.5%\n\nResult: Successfully processed the provided content with high accuracy.`

      case 'rag':
        return `Search results for: "${input}"\n\n📚 Found 8 relevant documents\n🔍 Hybrid search score: 0.89\n\nTop results:\n1. API Authentication Guide (95% relevance)\n2. Security Best Practices (92% relevance)\n3. Implementation Examples (88% relevance)\n\nGenerated response based on retrieved context with high confidence.`

      case 'streaming':
        return `Streaming analysis initiated:\n\n📊 Processing real-time data\n⚡ Event rate: 1.2k/sec\n🎯 Pattern detection: Active\n\nDetected patterns:\n- Normal traffic: 85%\n- Anomalies detected: 3\n- Alert threshold: Not exceeded\n\nContinuous monitoring active.`

      default:
        return `Standard processing completed:\n\n📝 Text analysis: Complete\n🧠 AI processing: Successful\n📊 Confidence: 96.2%\n\nGenerated comprehensive response based on the provided input with high accuracy and relevance.`
    }
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    setUploadedFiles(prev => [...prev, ...files])
  }

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const loadExample = (prompt: string) => {
    setInput(prompt)
  }

  const getStatusIcon = (status: PlaygroundRequest['status']) => {
    switch (status) {
      case 'running':
        return <Clock className="h-4 w-4 text-blue-500 animate-spin" />
      case 'completed':
        return <Activity className="h-4 w-4 text-green-500" />
      case 'error':
        return <Activity className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-400" />
    }
  }

  const selectedAgentInfo = agentTypes.find(agent => agent.value === selectedAgent)

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Agent Playground</h1>
          <p className="text-muted-foreground">
            Test and experiment with integrated agents
          </p>
        </div>
:start_line:195
-------
        <div className="flex space-x-2">
          <Button variant="outline" onClick={() => { /* TODO: Implement configuration modal */ }}>
            <Settings className="mr-2 h-4 w-4" />
            Configure
          </Button>
          <Button variant="destructive" onClick={async () => {
            if (confirm("Are you sure you want to clear all sessions? This action cannot be undone.")) {
              try {
                const response = await fetch('/v1/playground/clear_sessions', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': 'YOUR_API_KEY' // Replace with actual API key handling
                  },
                });
                if (response.ok) {
                  setRequests([]); // Clear local requests on successful API call
                  alert("All sessions cleared successfully!");
                } else {
                  const errorData = await response.json();
                  alert(`Failed to clear sessions: ${errorData.detail || response.statusText}`);
                }
              } catch (error: any) {
                alert(`Error clearing sessions: ${error.message}`);
              }
            }
          }}>
            <RotateCcw className="mr-2 h-4 w-4" />
            Clear All Sessions
          </Button>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Input Panel */}
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Agent Selection</CardTitle>
              <CardDescription>Choose the type of agent to test</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Select value={selectedAgent} onValueChange={setSelectedAgent}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {agentTypes.map((agent) => (
                    <SelectItem key={agent.value} value={agent.value}>
                      <div className="flex items-center space-x-2">
                        {agent.icon}
                        <span>{agent.label}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {selectedAgentInfo && (
                <div className="p-3 bg-accent rounded-md">
                  <div className="flex items-center space-x-2 mb-2">
                    {selectedAgentInfo.icon}
                    <span className="font-medium">{selectedAgentInfo.label}</span>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">
                    {selectedAgentInfo.description}
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {selectedAgentInfo.capabilities.map((capability) => (
                      <Badge key={capability} variant="secondary" className="text-xs">
                        {capability}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Input</CardTitle>
              <CardDescription>Enter your request or upload files</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <TextArea
                placeholder="Enter your request here..."
                value={input}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setInput(e.target.value)}
              />

              {/* File Upload */}
              {(selectedAgent === 'multimodal') && (
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm" asChild>
                      <label htmlFor="file-upload" className="cursor-pointer">
                        <Upload className="mr-2 h-4 w-4" />
                        Upload Files
                      </label>
                    </Button>
                    <input
                      id="file-upload"
                      type="file"
                      multiple
                      accept="image/*,audio/*"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </div>

                  {uploadedFiles.length > 0 && (
                    <div className="space-y-1">
                      {uploadedFiles.map((file, index) => (
                        <div key={index} className="flex items-center justify-between p-2 bg-accent rounded text-sm">
                          <span>{file.name}</span>
                          <Button variant="ghost" size="sm" onClick={() => removeFile(index)}>
                            ×
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div className="flex space-x-2">
                <Button
                  onClick={handleSubmit}
                  disabled={!input.trim() || isRunning}
                  className="flex-1"
                >
                  {isRunning ? (
                    <>
                      <Square className="mr-2 h-4 w-4" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Run Agent
                    </>
                  )}
                </Button>
                <Button variant="outline" onClick={() => setInput('')}>
                  <RotateCcw className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Examples Panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Example Prompts</CardTitle>
              <CardDescription>Try these example requests</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {examplePrompts[selectedAgent as keyof typeof examplePrompts]?.map((prompt, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  className="w-full text-left justify-start h-auto p-3"
                  onClick={() => loadExample(prompt)}
                >
                  <span className="text-sm">{prompt}</span>
                </Button>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Results */}
      {requests.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Results</CardTitle>
            <CardDescription>Agent responses and execution history</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {requests.map((request) => (
                <div key={request.id} className="border rounded-lg p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(request.status)}
                      <Badge variant="outline">{request.agentType}</Badge>
                      <span className="text-sm text-muted-foreground">
                        {new Date(request.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    {request.duration && (
                      <span className="text-sm text-muted-foreground">
                        {request.duration.toFixed(1)}s
                      </span>
                    )}
                  </div>

                  <div className="space-y-2">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Input:</p>
                      <p className="text-sm bg-accent p-2 rounded">{request.input}</p>
                    </div>

                    {request.response && (
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Response:</p>
                        <pre className="text-sm bg-accent p-2 rounded whitespace-pre-wrap">
                          {request.response}
                        </pre>
                      </div>
                    )}

                    {request.error && (
                      <div>
                        <p className="text-sm font-medium text-red-600">Error:</p>
                        <p className="text-sm bg-red-50 text-red-700 p-2 rounded">{request.error}</p>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
