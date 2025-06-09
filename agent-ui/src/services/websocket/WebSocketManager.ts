/**
 * WebSocket Manager for handling multiple WebSocket connections
 * with automatic reconnection, subscription management, and error handling
 */

export interface WebSocketConfig {
  url: string
  protocols?: string[]
  reconnectInterval?: number
  maxReconnectAttempts?: number
  heartbeatInterval?: number
  timeout?: number
}

export interface Subscription {
  id: string
  channel: string
  symbol?: string
  callback: (data: any) => void
}

export enum WebSocketState {
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  DISCONNECTED = 'DISCONNECTED',
  RECONNECTING = 'RECONNECTING',
  ERROR = 'ERROR'
}

export class WebSocketManager {
  private ws: WebSocket | null = null
  private config: Required<WebSocketConfig>
  private subscriptions = new Map<string, Subscription>()
  private reconnectAttempts = 0
  private reconnectTimer: NodeJS.Timeout | null = null
  private heartbeatTimer: NodeJS.Timeout | null = null
  private state: WebSocketState = WebSocketState.DISCONNECTED
  private listeners = new Map<string, Set<(data: any) => void>>()

  constructor(config: WebSocketConfig) {
    this.config = {
      protocols: [],
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      timeout: 10000,
      ...config
    }
  }

  /**
   * Connect to WebSocket server
   */
  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return
    }

    this.setState(WebSocketState.CONNECTING)

    try {
      this.ws = new WebSocket(this.config.url, this.config.protocols)
      this.setupEventHandlers()

      // Wait for connection with timeout
      await this.waitForConnection()
      this.setState(WebSocketState.CONNECTED)
      this.reconnectAttempts = 0
      this.startHeartbeat()
      this.resubscribeAll()

    } catch (error) {
      this.setState(WebSocketState.ERROR)
      this.handleReconnect()
      throw error
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.clearTimers()
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect')
      this.ws = null
    }
    
    this.setState(WebSocketState.DISCONNECTED)
  }

  /**
   * Subscribe to a channel/symbol
   */
  subscribe(subscription: Subscription): void {
    this.subscriptions.set(subscription.id, subscription)
    
    if (this.isConnected()) {
      this.sendSubscription(subscription)
    }
  }

  /**
   * Unsubscribe from a channel/symbol
   */
  unsubscribe(subscriptionId: string): void {
    const subscription = this.subscriptions.get(subscriptionId)
    if (subscription && this.isConnected()) {
      this.sendUnsubscription(subscription)
    }
    this.subscriptions.delete(subscriptionId)
  }

  /**
   * Send message to WebSocket server
   */
  send(message: any): void {
    if (this.isConnected() && this.ws) {
      this.ws.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket not connected, message not sent:', message)
    }
  }

  /**
   * Add event listener
   */
  addEventListener(event: string, callback: (data: any) => void): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set())
    }
    this.listeners.get(event)!.add(callback)
  }

  /**
   * Remove event listener
   */
  removeEventListener(event: string, callback: (data: any) => void): void {
    this.listeners.get(event)?.delete(callback)
  }

  /**
   * Get current connection state
   */
  getState(): WebSocketState {
    return this.state
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  private setupEventHandlers(): void {
    if (!this.ws) return

    this.ws.onopen = () => {
      console.log('WebSocket connected to:', this.config.url)
      this.emit('open', null)
    }

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        this.handleMessage(data)
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }

    this.ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason)
      this.setState(WebSocketState.DISCONNECTED)
      this.clearTimers()
      this.emit('close', { code: event.code, reason: event.reason })
      
      if (event.code !== 1000) { // Not a normal closure
        this.handleReconnect()
      }
    }

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      this.setState(WebSocketState.ERROR)
      this.emit('error', error)
    }
  }

  private async waitForConnection(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.ws) {
        reject(new Error('WebSocket not initialized'))
        return
      }

      const timeout = setTimeout(() => {
        reject(new Error('WebSocket connection timeout'))
      }, this.config.timeout)

      this.ws.onopen = () => {
        clearTimeout(timeout)
        resolve()
      }

      this.ws.onerror = (error) => {
        clearTimeout(timeout)
        reject(error)
      }
    })
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached')
      this.setState(WebSocketState.ERROR)
      return
    }

    this.setState(WebSocketState.RECONNECTING)
    this.reconnectAttempts++

    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.config.maxReconnectAttempts})`)

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(console.error)
    }, this.config.reconnectInterval)
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected()) {
        this.send({ type: 'ping', timestamp: Date.now() })
      }
    }, this.config.heartbeatInterval)
  }

  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  private setState(state: WebSocketState): void {
    this.state = state
    this.emit('stateChange', state)
  }

  private emit(event: string, data: any): void {
    this.listeners.get(event)?.forEach(callback => {
      try {
        callback(data)
      } catch (error) {
        console.error('Error in event listener:', error)
      }
    })
  }

  private handleMessage(data: any): void {
    // Handle heartbeat response
    if (data.type === 'pong') {
      return
    }

    // Route message to appropriate subscription
    const subscriptionId = this.getSubscriptionId(data)
    if (subscriptionId) {
      const subscription = this.subscriptions.get(subscriptionId)
      if (subscription) {
        subscription.callback(data)
      }
    }

    // Emit general message event
    this.emit('message', data)
  }

  private getSubscriptionId(data: any): string | null {
    // This should be implemented based on the specific WebSocket protocol
    // For now, return a generic ID based on channel and symbol
    if (data.channel && data.symbol) {
      return `${data.channel}:${data.symbol}`
    }
    return data.channel || null
  }

  private sendSubscription(subscription: Subscription): void {
    const message = {
      type: 'subscribe',
      channel: subscription.channel,
      symbol: subscription.symbol
    }
    this.send(message)
  }

  private sendUnsubscription(subscription: Subscription): void {
    const message = {
      type: 'unsubscribe',
      channel: subscription.channel,
      symbol: subscription.symbol
    }
    this.send(message)
  }

  private resubscribeAll(): void {
    this.subscriptions.forEach(subscription => {
      this.sendSubscription(subscription)
    })
  }
}
