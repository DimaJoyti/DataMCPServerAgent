'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  MessageSquare, 
  Bot, 
  Play, 
  Settings, 
  BarChart3,
  Globe,
  Smartphone,
  Mail,
  MessageCircle,
  ArrowLeft
} from 'lucide-react'

import { ChatInterface } from './ChatInterface'

// Types
interface BrandAgent {
  id: string
  name: string
  type: string
  description: string
  isActive: boolean
  deploymentChannels: string[]
}

interface ChatTesterProps {
  agents: BrandAgent[]
  onBack?: () => void
}

const channelIcons = {
  website_chat: MessageSquare,
  mobile_app: Smartphone,
  email: Mail,
  social_media: Globe,
  messaging_platform: MessageCircle,
}

const channelLabels = {
  website_chat: 'Website Chat',
  mobile_app: 'Mobile App',
  email: 'Email',
  social_media: 'Social Media',
  messaging_platform: 'Messaging',
}

export const ChatTester: React.FC<ChatTesterProps> = ({ agents, onBack }) => {
  const [selectedAgent, setSelectedAgent] = useState<BrandAgent | null>(null)
  const [selectedChannel, setSelectedChannel] = useState<string>('')
  const [chatActive, setChatActive] = useState(false)
  const [testResults, setTestResults] = useState<any[]>([])

  const activeAgents = agents.filter(agent => agent.isActive && agent.deploymentChannels.length > 0)

  const handleStartChat = () => {
    if (selectedAgent && selectedChannel) {
      setChatActive(true)
    }
  }

  const handleChatEnd = (conversationId: string, rating?: number) => {
    setChatActive(false)
    
    // Add to test results
    const result = {
      id: conversationId,
      agentId: selectedAgent?.id,
      agentName: selectedAgent?.name,
      channel: selectedChannel,
      rating: rating,
      timestamp: new Date().toISOString(),
    }
    
    setTestResults(prev => [result, ...prev])
  }

  const handleBackToSelection = () => {
    setChatActive(false)
    setSelectedAgent(null)
    setSelectedChannel('')
  }

  if (chatActive && selectedAgent) {
    return (
      <div className="h-full flex flex-col">
        <div className="mb-4">
          <Button
            variant="outline"
            onClick={handleBackToSelection}
            className="mb-4"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Agent Selection
          </Button>
        </div>
        
        <div className="flex-1">
          <ChatInterface
            agentId={selectedAgent.id}
            agentName={selectedAgent.name}
            channel={selectedChannel}
            userId="test-user"
            onConversationEnd={handleChatEnd}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">Chat Tester</h1>
          <p className="text-muted-foreground">
            Test your brand agents in real-time conversations
          </p>
        </div>
        {onBack && (
          <Button variant="outline" onClick={onBack}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Button>
        )}
      </div>

      <Tabs defaultValue="test" className="space-y-6">
        <TabsList>
          <TabsTrigger value="test">Test Chat</TabsTrigger>
          <TabsTrigger value="results">Test Results</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="test" className="space-y-6">
          {/* Agent Selection */}
          <Card>
            <CardHeader>
              <CardTitle>Select Agent to Test</CardTitle>
              <CardDescription>
                Choose an active agent that's deployed to at least one channel
              </CardDescription>
            </CardHeader>
            <CardContent>
              {activeAgents.length === 0 ? (
                <div className="text-center py-8">
                  <Bot className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No Active Agents</h3>
                  <p className="text-muted-foreground mb-4">
                    You need at least one active agent deployed to a channel to start testing.
                  </p>
                  <Button onClick={onBack}>
                    Go to Dashboard
                  </Button>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {activeAgents.map((agent) => (
                    <Card
                      key={agent.id}
                      className={`cursor-pointer transition-all duration-200 ${
                        selectedAgent?.id === agent.id
                          ? 'ring-2 ring-blue-500 bg-blue-50 dark:bg-blue-950'
                          : 'hover:shadow-md'
                      }`}
                      onClick={() => setSelectedAgent(agent)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-start space-x-3">
                          <div className="w-10 h-10 rounded-lg bg-blue-500 flex items-center justify-center">
                            <Bot className="h-5 w-5 text-white" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <h4 className="font-semibold truncate">{agent.name}</h4>
                            <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
                              {agent.description}
                            </p>
                            <div className="flex flex-wrap gap-1">
                              {agent.deploymentChannels.slice(0, 2).map((channel) => (
                                <Badge key={channel} variant="outline" className="text-xs">
                                  {channelLabels[channel as keyof typeof channelLabels] || channel}
                                </Badge>
                              ))}
                              {agent.deploymentChannels.length > 2 && (
                                <Badge variant="outline" className="text-xs">
                                  +{agent.deploymentChannels.length - 2}
                                </Badge>
                              )}
                            </div>
                          </div>
                          {selectedAgent?.id === agent.id && (
                            <div className="w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center">
                              <div className="w-2 h-2 bg-white rounded-full" />
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Channel Selection */}
          {selectedAgent && (
            <Card>
              <CardHeader>
                <CardTitle>Select Channel</CardTitle>
                <CardDescription>
                  Choose the channel where you want to test the conversation
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                  {selectedAgent.deploymentChannels.map((channel) => {
                    const Icon = channelIcons[channel as keyof typeof channelIcons] || MessageSquare
                    const label = channelLabels[channel as keyof typeof channelLabels] || channel
                    
                    return (
                      <Card
                        key={channel}
                        className={`cursor-pointer transition-all duration-200 ${
                          selectedChannel === channel
                            ? 'ring-2 ring-blue-500 bg-blue-50 dark:bg-blue-950'
                            : 'hover:shadow-md'
                        }`}
                        onClick={() => setSelectedChannel(channel)}
                      >
                        <CardContent className="p-4 text-center">
                          <Icon className="h-8 w-8 mx-auto mb-2 text-blue-500" />
                          <p className="text-sm font-medium">{label}</p>
                          {selectedChannel === channel && (
                            <div className="w-4 h-4 bg-blue-500 rounded-full mx-auto mt-2 flex items-center justify-center">
                              <div className="w-1.5 h-1.5 bg-white rounded-full" />
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Start Chat */}
          {selectedAgent && selectedChannel && (
            <Card>
              <CardContent className="p-6">
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                    <Play className="h-8 w-8 text-green-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold mb-2">Ready to Start</h3>
                    <p className="text-muted-foreground">
                      Test conversation with <strong>{selectedAgent.name}</strong> via{' '}
                      <strong>{channelLabels[selectedChannel as keyof typeof channelLabels]}</strong>
                    </p>
                  </div>
                  <Button onClick={handleStartChat} size="lg" className="bg-green-500 hover:bg-green-600">
                    <MessageSquare className="mr-2 h-5 w-5" />
                    Start Chat Test
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="results" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Test Results</CardTitle>
              <CardDescription>
                History of your chat tests and their outcomes
              </CardDescription>
            </CardHeader>
            <CardContent>
              {testResults.length === 0 ? (
                <div className="text-center py-8">
                  <BarChart3 className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No Test Results</h3>
                  <p className="text-muted-foreground">
                    Start testing your agents to see results here.
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {testResults.map((result) => (
                    <div
                      key={result.id}
                      className="flex items-center justify-between p-4 border rounded-lg"
                    >
                      <div className="flex items-center space-x-4">
                        <div className="w-10 h-10 rounded-lg bg-blue-500 flex items-center justify-center">
                          <Bot className="h-5 w-5 text-white" />
                        </div>
                        <div>
                          <h4 className="font-medium">{result.agentName}</h4>
                          <p className="text-sm text-muted-foreground">
                            {channelLabels[result.channel as keyof typeof channelLabels]} •{' '}
                            {new Date(result.timestamp).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-4">
                        {result.rating && (
                          <div className="flex items-center space-x-1">
                            <span className="text-sm font-medium">{result.rating}/5</span>
                            <div className="flex">
                              {[...Array(5)].map((_, i) => (
                                <div
                                  key={i}
                                  className={`w-3 h-3 ${
                                    i < result.rating ? 'text-yellow-500' : 'text-gray-300'
                                  }`}
                                >
                                  ⭐
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        <Badge variant="outline">Completed</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Test Settings</CardTitle>
              <CardDescription>
                Configure testing parameters and preferences
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Default User ID
                  </label>
                  <input
                    type="text"
                    defaultValue="test-user"
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Test Environment
                  </label>
                  <select className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                    <option>Development</option>
                    <option>Staging</option>
                    <option>Production</option>
                  </select>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium">Auto-save Test Results</h4>
                  <p className="text-sm text-muted-foreground">
                    Automatically save conversation logs and metrics
                  </p>
                </div>
                <input type="checkbox" defaultChecked className="w-4 h-4" />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium">Enable Debug Mode</h4>
                  <p className="text-sm text-muted-foreground">
                    Show detailed logs and response times
                  </p>
                </div>
                <input type="checkbox" className="w-4 h-4" />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
