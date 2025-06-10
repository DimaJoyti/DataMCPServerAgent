/**
 * WebSocket service for real-time Infinite Agentic Loop updates
 */

import { type WebSocketMessage, type InfiniteLoopError } from '@/types/infinite-loop'
import { infiniteLoopAPI } from './api'

export type WebSocketEventHandler = (message: WebSocketMessage) => void
export type WebSocketErrorHandler = (error: InfiniteLoopError) => void
export type WebSocketConnectionHandler = (connected: boolean) => void

class InfiniteLoopWebSocketService {
  private ws: WebSocket | null = null
  private sessionId: string | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private isManuallyDisconnected = false

  // Event handlers
  private messageHandlers: WebSocketEventHandler[] = []
  private errorHandlers: WebSocketErrorHandler[] = []
  private connectionHandlers: WebSocketConnectionHandler[] = []

  /**
   * Connect to WebSocket for a specific session
   */
  connect(sessionId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.sessionId = sessionId
        this.isManuallyDisconnected = false
        
        const wsUrl = infiniteLoopAPI.getWebSocketUrl(sessionId)
        this.ws = new WebSocket(wsUrl)

        this.ws.onopen = () => {
          console.log(`WebSocket connected for session: ${sessionId}`)
          this.reconnectAttempts = 0
          this.notifyConnectionHandlers(true)
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data)
            this.notifyMessageHandlers(message)
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error)
            this.notifyErrorHandlers({
              code: 'PARSE_ERROR',
              message: 'Failed to parse WebSocket message',
              details: error,
              timestamp: new Date().toISOString(),
              sessionId
            })
          }
        }

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          this.notifyErrorHandlers({
            code: 'CONNECTION_ERROR',
            message: 'WebSocket connection error',
            details: error,
            timestamp: new Date().toISOString(),
            sessionId
          })
          reject(error)
        }

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason)
          this.notifyConnectionHandlers(false)
          
          if (!this.isManuallyDisconnected && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect()
          }
        }

      } catch (error) {
        reject(error)
      }
    })
  }

  /**
   * Disconnect WebSocket
   */
  disconnect(): void {
    this.isManuallyDisconnected = true
    
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    
    this.sessionId = null
    this.reconnectAttempts = 0
    this.notifyConnectionHandlers(false)
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  /**
   * Get current session ID
   */
  getCurrentSessionId(): string | null {
    return this.sessionId
  }

  /**
   * Send message to WebSocket
   */
  send(message: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket is not connected. Cannot send message.')
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    if (this.sessionId && !this.isManuallyDisconnected) {
      this.reconnectAttempts++
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
      
      console.log(`Scheduling WebSocket reconnect attempt ${this.reconnectAttempts} in ${delay}ms`)
      
      setTimeout(() => {
        if (this.sessionId && !this.isManuallyDisconnected) {
          this.connect(this.sessionId).catch((error) => {
            console.error('WebSocket reconnect failed:', error)
          })
        }
      }, delay)
    }
  }

  /**
   * Add message event handler
   */
  onMessage(handler: WebSocketEventHandler): () => void {
    this.messageHandlers.push(handler)
    
    // Return unsubscribe function
    return () => {
      const index = this.messageHandlers.indexOf(handler)
      if (index > -1) {
        this.messageHandlers.splice(index, 1)
      }
    }
  }

  /**
   * Add error event handler
   */
  onError(handler: WebSocketErrorHandler): () => void {
    this.errorHandlers.push(handler)
    
    // Return unsubscribe function
    return () => {
      const index = this.errorHandlers.indexOf(handler)
      if (index > -1) {
        this.errorHandlers.splice(index, 1)
      }
    }
  }

  /**
   * Add connection event handler
   */
  onConnection(handler: WebSocketConnectionHandler): () => void {
    this.connectionHandlers.push(handler)
    
    // Return unsubscribe function
    return () => {
      const index = this.connectionHandlers.indexOf(handler)
      if (index > -1) {
        this.connectionHandlers.splice(index, 1)
      }
    }
  }

  /**
   * Notify all message handlers
   */
  private notifyMessageHandlers(message: WebSocketMessage): void {
    this.messageHandlers.forEach(handler => {
      try {
        handler(message)
      } catch (error) {
        console.error('Error in message handler:', error)
      }
    })
  }

  /**
   * Notify all error handlers
   */
  private notifyErrorHandlers(error: InfiniteLoopError): void {
    this.errorHandlers.forEach(handler => {
      try {
        handler(error)
      } catch (handlerError) {
        console.error('Error in error handler:', handlerError)
      }
    })
  }

  /**
   * Notify all connection handlers
   */
  private notifyConnectionHandlers(connected: boolean): void {
    this.connectionHandlers.forEach(handler => {
      try {
        handler(connected)
      } catch (error) {
        console.error('Error in connection handler:', error)
      }
    })
  }

  /**
   * Clean up all handlers
   */
  cleanup(): void {
    this.disconnect()
    this.messageHandlers = []
    this.errorHandlers = []
    this.connectionHandlers = []
  }
}

// Export singleton instance
export const infiniteLoopWebSocket = new InfiniteLoopWebSocketService()
export default infiniteLoopWebSocket
