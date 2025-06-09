'use client'

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { 
  Send, 
  Bot, 
  User, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  Loader2,
  Star,
  ThumbsUp,
  ThumbsDown,
  MoreVertical
} from 'lucide-react'

// Types
interface Message {
  id: string
  sender_type: 'user' | 'agent' | 'system'
  content: string
  timestamp: string
  status?: 'pending' | 'sent' | 'delivered' | 'read' | 'failed'
  response_time_ms?: number
  knowledge_sources?: string[]
}

interface ConversationStatus {
  id: string
  status: 'active' | 'waiting' | 'escalated' | 'resolved' | 'closed'
  started_at: string
  duration_seconds: number
  message_count: number
  last_activity: string
}

interface ChatInterfaceProps {
  agentId: string
  agentName: string
  channel: string
  userId?: string
  onConversationEnd?: (conversationId: string, rating?: number) => void
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  agentId,
  agentName,
  channel,
  userId,
  onConversationEnd
}) => {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isConnected, setIsConnected] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const [agentTyping, setAgentTyping] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [conversationStatus, setConversationStatus] = useState<ConversationStatus | null>(null)
  const [showRating, setShowRating] = useState(false)
  const [rating, setRating] = useState<number | null>(null)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const websocketRef = useRef<WebSocket | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Start conversation and connect WebSocket
  const startConversation = useCallback(async () => {
    try {
      // Start conversation via API
      const response = await fetch('/api/v1/brand-agents/live-conversations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          brand_agent_id: agentId,
          channel: channel,
          user_id: userId,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to start conversation')
      }

      const conversation = await response.json()
      setConversationId(conversation.id)

      // Connect WebSocket
      const wsUrl = `ws://localhost:8000/ws/chat/${conversation.id}?session_token=${conversation.session_token}`
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        setIsConnected(true)
        console.log('WebSocket connected')
      }

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data)
        handleWebSocketMessage(message)
      }

      ws.onclose = () => {
        setIsConnected(false)
        console.log('WebSocket disconnected')
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setIsConnected(false)
      }

      websocketRef.current = ws

    } catch (error) {
      console.error('Failed to start conversation:', error)
    }
  }, [agentId, channel, userId])

  // Handle WebSocket messages
  const handleWebSocketMessage = (message: any) => {
    switch (message.type) {
      case 'connection_established':
        console.log('Connection established:', message.data)
        break

      case 'message_received':
        const newMessage: Message = {
          id: message.data.message_id,
          sender_type: message.data.sender_type,
          content: message.data.content,
          timestamp: message.data.timestamp,
          status: message.data.status,
          response_time_ms: message.data.response_time_ms,
          knowledge_sources: message.data.knowledge_sources,
        }
        setMessages(prev => [...prev, newMessage])
        break

      case 'agent_typing':
        setAgentTyping(message.data.is_typing)
        break

      case 'user_typing':
        // Handle other users typing (for multi-user chats)
        break

      case 'conversation_status':
        setConversationStatus(message.data)
        break

      case 'conversation_ended':
        setShowRating(true)
        setIsConnected(false)
        break

      case 'system_message':
        const systemMessage: Message = {
          id: `system-${Date.now()}`,
          sender_type: 'system',
          content: message.data.message,
          timestamp: message.data.timestamp,
        }
        setMessages(prev => [...prev, systemMessage])
        break

      case 'error':
        console.error('WebSocket error:', message.data.message)
        break

      default:
        console.log('Unknown message type:', message.type)
    }
  }

  // Send message
  const sendMessage = () => {
    if (!inputMessage.trim() || !websocketRef.current || !isConnected) return

    const message = {
      type: 'user_message',
      data: {
        content: inputMessage.trim(),
        message_type: 'text',
        metadata: {
          timestamp: new Date().toISOString(),
        },
      },
    }

    websocketRef.current.send(JSON.stringify(message))
    setInputMessage('')
    setIsTyping(false)
  }

  // Handle typing indicator
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputMessage(e.target.value)

    if (!isTyping && websocketRef.current) {
      setIsTyping(true)
      websocketRef.current.send(JSON.stringify({
        type: 'typing_start',
        data: { user_id: userId }
      }))
    }

    // Clear existing timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current)
    }

    // Set new timeout
    typingTimeoutRef.current = setTimeout(() => {
      if (isTyping && websocketRef.current) {
        setIsTyping(false)
        websocketRef.current.send(JSON.stringify({
          type: 'typing_stop',
          data: { user_id: userId }
        }))
      }
    }, 1000)
  }

  // End conversation
  const endConversation = (satisfactionRating?: number) => {
    if (websocketRef.current && conversationId) {
      websocketRef.current.send(JSON.stringify({
        type: 'end_conversation',
        data: {
          reason: 'user_ended',
          satisfaction_rating: satisfactionRating,
        },
      }))
    }

    if (onConversationEnd && conversationId) {
      onConversationEnd(conversationId, satisfactionRating)
    }

    setShowRating(false)
  }

  // Start conversation on mount
  useEffect(() => {
    startConversation()

    return () => {
      if (websocketRef.current) {
        websocketRef.current.close()
      }
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current)
      }
    }
  }, [startConversation])

  // Handle Enter key
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // Format timestamp
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }

  // Get status icon
  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'sent':
        return <CheckCircle className="h-3 w-3 text-blue-500" />
      case 'delivered':
        return <CheckCircle className="h-3 w-3 text-green-500" />
      case 'read':
        return <CheckCircle className="h-3 w-3 text-green-600" />
      case 'failed':
        return <AlertCircle className="h-3 w-3 text-red-500" />
      default:
        return <Clock className="h-3 w-3 text-gray-400" />
    }
  }

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto">
      {/* Header */}
      <Card className="mb-4">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center">
                <Bot className="h-6 w-6 text-white" />
              </div>
              <div>
                <CardTitle className="text-lg">{agentName}</CardTitle>
                <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                  <Badge variant={isConnected ? "default" : "secondary"}>
                    {isConnected ? "Online" : "Offline"}
                  </Badge>
                  <span>•</span>
                  <span>{channel.replace('_', ' ')}</span>
                  {conversationStatus && (
                    <>
                      <span>•</span>
                      <span>{Math.floor(conversationStatus.duration_seconds / 60)}m</span>
                    </>
                  )}
                </div>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowRating(true)}
              disabled={!isConnected}
            >
              <MoreVertical className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
      </Card>

      {/* Messages */}
      <Card className="flex-1 flex flex-col">
        <CardContent className="flex-1 p-4 overflow-y-auto">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.sender_type === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                    message.sender_type === 'user'
                      ? 'bg-blue-500 text-white'
                      : message.sender_type === 'system'
                      ? 'bg-gray-100 text-gray-700 text-center'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <div className="flex items-start space-x-2">
                    {message.sender_type === 'agent' && (
                      <Bot className="h-4 w-4 mt-1 text-blue-500" />
                    )}
                    {message.sender_type === 'user' && (
                      <User className="h-4 w-4 mt-1 text-white" />
                    )}
                    <div className="flex-1">
                      <p className="text-sm">{message.content}</p>
                      <div className="flex items-center justify-between mt-1">
                        <span className="text-xs opacity-70">
                          {formatTime(message.timestamp)}
                        </span>
                        <div className="flex items-center space-x-1">
                          {message.response_time_ms && (
                            <span className="text-xs opacity-70">
                              {message.response_time_ms}ms
                            </span>
                          )}
                          {message.sender_type === 'user' && getStatusIcon(message.status)}
                        </div>
                      </div>
                      {message.knowledge_sources && message.knowledge_sources.length > 0 && (
                        <div className="mt-1">
                          <Badge variant="outline" className="text-xs">
                            {message.knowledge_sources.length} sources
                          </Badge>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* Typing indicator */}
            {agentTyping && (
              <div className="flex justify-start">
                <div className="bg-gray-100 px-4 py-2 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Bot className="h-4 w-4 text-blue-500" />
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </CardContent>

        {/* Input */}
        <div className="border-t p-4">
          <div className="flex space-x-2">
            <input
              ref={inputRef}
              type="text"
              value={inputMessage}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              disabled={!isConnected}
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            />
            <Button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || !isConnected}
              className="px-4"
            >
              {isConnected ? (
                <Send className="h-4 w-4" />
              ) : (
                <Loader2 className="h-4 w-4 animate-spin" />
              )}
            </Button>
          </div>
          {isTyping && (
            <p className="text-xs text-muted-foreground mt-1">Typing...</p>
          )}
        </div>
      </Card>

      {/* Rating Modal */}
      {showRating && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <Card className="w-96">
            <CardHeader>
              <CardTitle>Rate Your Experience</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                How was your conversation with {agentName}?
              </p>
              <div className="flex justify-center space-x-2">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    onClick={() => setRating(star)}
                    className={`p-1 ${
                      rating && star <= rating ? 'text-yellow-500' : 'text-gray-300'
                    }`}
                  >
                    <Star className="h-6 w-6 fill-current" />
                  </button>
                ))}
              </div>
              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  onClick={() => endConversation()}
                  className="flex-1"
                >
                  Skip
                </Button>
                <Button
                  onClick={() => endConversation(rating || undefined)}
                  disabled={!rating}
                  className="flex-1"
                >
                  Submit
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
