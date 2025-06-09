/**
 * Market Data WebSocket Service
 * Handles real-time market data feeds including quotes, trades, and order book updates
 */

import { WebSocketManager, WebSocketConfig, Subscription } from './WebSocketManager'

export interface Quote {
  symbol: string
  bid: number
  ask: number
  bidSize: number
  askSize: number
  timestamp: number
}

export interface Trade {
  symbol: string
  price: number
  size: number
  side: 'buy' | 'sell'
  timestamp: number
  tradeId: string
}

export interface OrderBookLevel {
  price: number
  size: number
  orders: number
}

export interface OrderBookUpdate {
  symbol: string
  bids: OrderBookLevel[]
  asks: OrderBookLevel[]
  timestamp: number
}

export interface Ticker {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  high24h: number
  low24h: number
  timestamp: number
}

export type MarketDataCallback = {
  onQuote?: (quote: Quote) => void
  onTrade?: (trade: Trade) => void
  onOrderBook?: (orderBook: OrderBookUpdate) => void
  onTicker?: (ticker: Ticker) => void
  onError?: (error: any) => void
}

export class MarketDataWebSocket {
  private wsManager: WebSocketManager
  private subscriptions = new Map<string, MarketDataCallback>()

  constructor(config: WebSocketConfig) {
    this.wsManager = new WebSocketManager(config)
    this.setupEventHandlers()
  }

  /**
   * Connect to market data feed
   */
  async connect(): Promise<void> {
    await this.wsManager.connect()
  }

  /**
   * Disconnect from market data feed
   */
  disconnect(): void {
    this.wsManager.disconnect()
  }

  /**
   * Subscribe to real-time quotes for a symbol
   */
  subscribeToQuotes(symbol: string, callback: (quote: Quote) => void): string {
    const subscriptionId = `quotes:${symbol}`
    
    this.wsManager.subscribe({
      id: subscriptionId,
      channel: 'quotes',
      symbol,
      callback: (data) => {
        if (data.type === 'quote') {
          callback(this.parseQuote(data))
        }
      }
    })

    return subscriptionId
  }

  /**
   * Subscribe to real-time trades for a symbol
   */
  subscribeToTrades(symbol: string, callback: (trade: Trade) => void): string {
    const subscriptionId = `trades:${symbol}`
    
    this.wsManager.subscribe({
      id: subscriptionId,
      channel: 'trades',
      symbol,
      callback: (data) => {
        if (data.type === 'trade') {
          callback(this.parseTrade(data))
        }
      }
    })

    return subscriptionId
  }

  /**
   * Subscribe to order book updates for a symbol
   */
  subscribeToOrderBook(symbol: string, callback: (orderBook: OrderBookUpdate) => void): string {
    const subscriptionId = `orderbook:${symbol}`
    
    this.wsManager.subscribe({
      id: subscriptionId,
      channel: 'orderbook',
      symbol,
      callback: (data) => {
        if (data.type === 'orderbook') {
          callback(this.parseOrderBook(data))
        }
      }
    })

    return subscriptionId
  }

  /**
   * Subscribe to ticker updates for a symbol
   */
  subscribeToTicker(symbol: string, callback: (ticker: Ticker) => void): string {
    const subscriptionId = `ticker:${symbol}`
    
    this.wsManager.subscribe({
      id: subscriptionId,
      channel: 'ticker',
      symbol,
      callback: (data) => {
        if (data.type === 'ticker') {
          callback(this.parseTicker(data))
        }
      }
    })

    return subscriptionId
  }

  /**
   * Subscribe to all market data for a symbol
   */
  subscribeToSymbol(symbol: string, callbacks: MarketDataCallback): string[] {
    const subscriptionIds: string[] = []

    if (callbacks.onQuote) {
      subscriptionIds.push(this.subscribeToQuotes(symbol, callbacks.onQuote))
    }

    if (callbacks.onTrade) {
      subscriptionIds.push(this.subscribeToTrades(symbol, callbacks.onTrade))
    }

    if (callbacks.onOrderBook) {
      subscriptionIds.push(this.subscribeToOrderBook(symbol, callbacks.onOrderBook))
    }

    if (callbacks.onTicker) {
      subscriptionIds.push(this.subscribeToTicker(symbol, callbacks.onTicker))
    }

    this.subscriptions.set(symbol, callbacks)
    return subscriptionIds
  }

  /**
   * Unsubscribe from a specific subscription
   */
  unsubscribe(subscriptionId: string): void {
    this.wsManager.unsubscribe(subscriptionId)
  }

  /**
   * Unsubscribe from all subscriptions for a symbol
   */
  unsubscribeFromSymbol(symbol: string): void {
    const channels = ['quotes', 'trades', 'orderbook', 'ticker']
    channels.forEach(channel => {
      this.wsManager.unsubscribe(`${channel}:${symbol}`)
    })
    this.subscriptions.delete(symbol)
  }

  /**
   * Get connection state
   */
  getConnectionState() {
    return this.wsManager.getState()
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.wsManager.isConnected()
  }

  /**
   * Add connection event listeners
   */
  addEventListener(event: string, callback: (data: any) => void): void {
    this.wsManager.addEventListener(event, callback)
  }

  /**
   * Remove connection event listeners
   */
  removeEventListener(event: string, callback: (data: any) => void): void {
    this.wsManager.removeEventListener(event, callback)
  }

  private setupEventHandlers(): void {
    this.wsManager.addEventListener('error', (error) => {
      console.error('Market data WebSocket error:', error)
      this.subscriptions.forEach(callbacks => {
        callbacks.onError?.(error)
      })
    })

    this.wsManager.addEventListener('close', (event) => {
      console.log('Market data WebSocket closed:', event)
    })

    this.wsManager.addEventListener('stateChange', (state) => {
      console.log('Market data WebSocket state changed:', state)
    })
  }

  private parseQuote(data: any): Quote {
    return {
      symbol: data.symbol,
      bid: parseFloat(data.bid),
      ask: parseFloat(data.ask),
      bidSize: parseFloat(data.bidSize || data.bid_size || 0),
      askSize: parseFloat(data.askSize || data.ask_size || 0),
      timestamp: data.timestamp || Date.now()
    }
  }

  private parseTrade(data: any): Trade {
    return {
      symbol: data.symbol,
      price: parseFloat(data.price),
      size: parseFloat(data.size || data.quantity),
      side: data.side || (data.type === 'buy' ? 'buy' : 'sell'),
      timestamp: data.timestamp || Date.now(),
      tradeId: data.tradeId || data.trade_id || data.id || String(Date.now())
    }
  }

  private parseOrderBook(data: any): OrderBookUpdate {
    const parseLevels = (levels: any[]): OrderBookLevel[] => {
      return levels.map(level => ({
        price: parseFloat(level.price || level[0]),
        size: parseFloat(level.size || level[1]),
        orders: parseInt(level.orders || level[2] || '1')
      }))
    }

    return {
      symbol: data.symbol,
      bids: parseLevels(data.bids || []),
      asks: parseLevels(data.asks || []),
      timestamp: data.timestamp || Date.now()
    }
  }

  private parseTicker(data: any): Ticker {
    return {
      symbol: data.symbol,
      price: parseFloat(data.price || data.last),
      change: parseFloat(data.change || 0),
      changePercent: parseFloat(data.changePercent || data.change_percent || 0),
      volume: parseFloat(data.volume || 0),
      high24h: parseFloat(data.high24h || data.high || 0),
      low24h: parseFloat(data.low24h || data.low || 0),
      timestamp: data.timestamp || Date.now()
    }
  }
}

// Factory function for different exchanges
export class MarketDataWebSocketFactory {
  static createBinanceWebSocket(): MarketDataWebSocket {
    return new MarketDataWebSocket({
      url: 'wss://stream.binance.com:9443/ws/btcusdt@ticker/btcusdt@depth/btcusdt@trade',
      reconnectInterval: 5000,
      maxReconnectAttempts: 10
    })
  }

  static createCoinbaseWebSocket(): MarketDataWebSocket {
    return new MarketDataWebSocket({
      url: 'wss://ws-feed.exchange.coinbase.com',
      reconnectInterval: 5000,
      maxReconnectAttempts: 10
    })
  }

  static createLocalWebSocket(port: number = 8765): MarketDataWebSocket {
    return new MarketDataWebSocket({
      url: `ws://localhost:${port}/ws/market-data`,
      reconnectInterval: 3000,
      maxReconnectAttempts: 20
    })
  }

  static createAlpacaWebSocket(): MarketDataWebSocket {
    return new MarketDataWebSocket({
      url: 'wss://stream.data.alpaca.markets/v2/iex',
      reconnectInterval: 5000,
      maxReconnectAttempts: 10
    })
  }
}
