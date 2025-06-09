/**
 * Trading WebSocket Service
 * Handles real-time trading operations, order updates, and position changes
 */

import { WebSocketManager, WebSocketConfig } from './WebSocketManager'

export interface OrderUpdate {
  orderId: string
  clientOrderId?: string
  symbol: string
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stop' | 'stop-limit'
  status: 'pending' | 'filled' | 'partially_filled' | 'cancelled' | 'rejected'
  quantity: number
  price?: number
  filledQuantity: number
  averageFillPrice?: number
  timestamp: number
  reason?: string
}

export interface PositionUpdate {
  symbol: string
  side: 'long' | 'short'
  quantity: number
  entryPrice: number
  currentPrice: number
  unrealizedPnL: number
  realizedPnL: number
  timestamp: number
}

export interface BalanceUpdate {
  currency: string
  available: number
  locked: number
  total: number
  timestamp: number
}

export interface TradeExecution {
  tradeId: string
  orderId: string
  symbol: string
  side: 'buy' | 'sell'
  quantity: number
  price: number
  commission: number
  timestamp: number
}

export interface RiskAlert {
  id: string
  type: 'position_limit' | 'loss_limit' | 'exposure_limit' | 'margin_call'
  severity: 'low' | 'medium' | 'high' | 'critical'
  message: string
  symbol?: string
  value?: number
  limit?: number
  timestamp: number
}

export type TradingCallback = {
  onOrderUpdate?: (order: OrderUpdate) => void
  onPositionUpdate?: (position: PositionUpdate) => void
  onBalanceUpdate?: (balance: BalanceUpdate) => void
  onTradeExecution?: (trade: TradeExecution) => void
  onRiskAlert?: (alert: RiskAlert) => void
  onError?: (error: any) => void
}

export class TradingWebSocket {
  private wsManager: WebSocketManager
  private callbacks: TradingCallback = {}
  private authenticated = false

  constructor(config: WebSocketConfig) {
    this.wsManager = new WebSocketManager(config)
    this.setupEventHandlers()
  }

  /**
   * Connect to trading WebSocket
   */
  async connect(): Promise<void> {
    await this.wsManager.connect()
  }

  /**
   * Disconnect from trading WebSocket
   */
  disconnect(): void {
    this.wsManager.disconnect()
    this.authenticated = false
  }

  /**
   * Authenticate with the trading system
   */
  async authenticate(apiKey: string, signature: string, timestamp: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const authMessage = {
        type: 'auth',
        apiKey,
        signature,
        timestamp
      }

      const timeoutId = setTimeout(() => {
        reject(new Error('Authentication timeout'))
      }, 10000)

      const handleAuthResponse = (data: any) => {
        if (data.type === 'auth_response') {
          clearTimeout(timeoutId)
          this.wsManager.removeEventListener('message', handleAuthResponse)
          
          if (data.success) {
            this.authenticated = true
            resolve()
          } else {
            reject(new Error(data.error || 'Authentication failed'))
          }
        }
      }

      this.wsManager.addEventListener('message', handleAuthResponse)
      this.wsManager.send(authMessage)
    })
  }

  /**
   * Set trading event callbacks
   */
  setCallbacks(callbacks: TradingCallback): void {
    this.callbacks = { ...this.callbacks, ...callbacks }
  }

  /**
   * Subscribe to order updates
   */
  subscribeToOrders(): void {
    if (!this.authenticated) {
      throw new Error('Must authenticate before subscribing to orders')
    }

    this.wsManager.subscribe({
      id: 'orders',
      channel: 'orders',
      callback: (data) => {
        if (data.type === 'order_update') {
          this.callbacks.onOrderUpdate?.(this.parseOrderUpdate(data))
        }
      }
    })
  }

  /**
   * Subscribe to position updates
   */
  subscribeToPositions(): void {
    if (!this.authenticated) {
      throw new Error('Must authenticate before subscribing to positions')
    }

    this.wsManager.subscribe({
      id: 'positions',
      channel: 'positions',
      callback: (data) => {
        if (data.type === 'position_update') {
          this.callbacks.onPositionUpdate?.(this.parsePositionUpdate(data))
        }
      }
    })
  }

  /**
   * Subscribe to balance updates
   */
  subscribeToBalances(): void {
    if (!this.authenticated) {
      throw new Error('Must authenticate before subscribing to balances')
    }

    this.wsManager.subscribe({
      id: 'balances',
      channel: 'balances',
      callback: (data) => {
        if (data.type === 'balance_update') {
          this.callbacks.onBalanceUpdate?.(this.parseBalanceUpdate(data))
        }
      }
    })
  }

  /**
   * Subscribe to trade executions
   */
  subscribeToTrades(): void {
    if (!this.authenticated) {
      throw new Error('Must authenticate before subscribing to trades')
    }

    this.wsManager.subscribe({
      id: 'trades',
      channel: 'trades',
      callback: (data) => {
        if (data.type === 'trade_execution') {
          this.callbacks.onTradeExecution?.(this.parseTradeExecution(data))
        }
      }
    })
  }

  /**
   * Subscribe to risk alerts
   */
  subscribeToRiskAlerts(): void {
    if (!this.authenticated) {
      throw new Error('Must authenticate before subscribing to risk alerts')
    }

    this.wsManager.subscribe({
      id: 'risk_alerts',
      channel: 'risk',
      callback: (data) => {
        if (data.type === 'risk_alert') {
          this.callbacks.onRiskAlert?.(this.parseRiskAlert(data))
        }
      }
    })
  }

  /**
   * Subscribe to all trading updates
   */
  subscribeToAll(): void {
    this.subscribeToOrders()
    this.subscribeToPositions()
    this.subscribeToBalances()
    this.subscribeToTrades()
    this.subscribeToRiskAlerts()
  }

  /**
   * Send order request
   */
  sendOrder(order: {
    symbol: string
    side: 'buy' | 'sell'
    type: 'market' | 'limit' | 'stop' | 'stop-limit'
    quantity: number
    price?: number
    stopPrice?: number
    clientOrderId?: string
  }): void {
    if (!this.authenticated) {
      throw new Error('Must authenticate before sending orders')
    }

    const orderMessage = {
      type: 'new_order',
      ...order,
      timestamp: Date.now()
    }

    this.wsManager.send(orderMessage)
  }

  /**
   * Cancel order
   */
  cancelOrder(orderId: string): void {
    if (!this.authenticated) {
      throw new Error('Must authenticate before cancelling orders')
    }

    const cancelMessage = {
      type: 'cancel_order',
      orderId,
      timestamp: Date.now()
    }

    this.wsManager.send(cancelMessage)
  }

  /**
   * Modify order
   */
  modifyOrder(orderId: string, updates: {
    quantity?: number
    price?: number
    stopPrice?: number
  }): void {
    if (!this.authenticated) {
      throw new Error('Must authenticate before modifying orders')
    }

    const modifyMessage = {
      type: 'modify_order',
      orderId,
      ...updates,
      timestamp: Date.now()
    }

    this.wsManager.send(modifyMessage)
  }

  /**
   * Get connection state
   */
  getConnectionState() {
    return this.wsManager.getState()
  }

  /**
   * Check if connected and authenticated
   */
  isReady(): boolean {
    return this.wsManager.isConnected() && this.authenticated
  }

  private setupEventHandlers(): void {
    this.wsManager.addEventListener('error', (error) => {
      console.error('Trading WebSocket error:', error)
      this.callbacks.onError?.(error)
    })

    this.wsManager.addEventListener('close', (event) => {
      console.log('Trading WebSocket closed:', event)
      this.authenticated = false
    })

    this.wsManager.addEventListener('stateChange', (state) => {
      console.log('Trading WebSocket state changed:', state)
    })
  }

  private parseOrderUpdate(data: any): OrderUpdate {
    return {
      orderId: data.orderId || data.order_id,
      clientOrderId: data.clientOrderId || data.client_order_id,
      symbol: data.symbol,
      side: data.side,
      type: data.type || data.order_type,
      status: data.status,
      quantity: parseFloat(data.quantity),
      price: data.price ? parseFloat(data.price) : undefined,
      filledQuantity: parseFloat(data.filledQuantity || data.filled_quantity || 0),
      averageFillPrice: data.averageFillPrice ? parseFloat(data.averageFillPrice) : undefined,
      timestamp: data.timestamp || Date.now(),
      reason: data.reason
    }
  }

  private parsePositionUpdate(data: any): PositionUpdate {
    return {
      symbol: data.symbol,
      side: data.side,
      quantity: parseFloat(data.quantity),
      entryPrice: parseFloat(data.entryPrice || data.entry_price),
      currentPrice: parseFloat(data.currentPrice || data.current_price),
      unrealizedPnL: parseFloat(data.unrealizedPnL || data.unrealized_pnl),
      realizedPnL: parseFloat(data.realizedPnL || data.realized_pnl),
      timestamp: data.timestamp || Date.now()
    }
  }

  private parseBalanceUpdate(data: any): BalanceUpdate {
    return {
      currency: data.currency,
      available: parseFloat(data.available),
      locked: parseFloat(data.locked),
      total: parseFloat(data.total),
      timestamp: data.timestamp || Date.now()
    }
  }

  private parseTradeExecution(data: any): TradeExecution {
    return {
      tradeId: data.tradeId || data.trade_id,
      orderId: data.orderId || data.order_id,
      symbol: data.symbol,
      side: data.side,
      quantity: parseFloat(data.quantity),
      price: parseFloat(data.price),
      commission: parseFloat(data.commission || 0),
      timestamp: data.timestamp || Date.now()
    }
  }

  private parseRiskAlert(data: any): RiskAlert {
    return {
      id: data.id,
      type: data.type,
      severity: data.severity,
      message: data.message,
      symbol: data.symbol,
      value: data.value ? parseFloat(data.value) : undefined,
      limit: data.limit ? parseFloat(data.limit) : undefined,
      timestamp: data.timestamp || Date.now()
    }
  }
}
