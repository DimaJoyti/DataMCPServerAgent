/**
 * Trading API Service
 * Handles all REST API calls for trading operations
 */

import {
  ApiResponse,
  CreateOrderRequest,
  Order,
  ModifyOrderRequest,
  Position,
  Portfolio,
  Balance,
  RiskMetrics,
  RiskLimits,
  PerformanceMetrics,
  SystemHealth,
  SystemComponent,
  AuthRequest,
  AuthResponse,
  PaginationParams,
  PaginatedResponse,
  OrderFilter,
  TradeFilter,
  TradingConfig,
  TradingStrategy,
  AlgorithmicOrderRequest
} from './types'

export class TradingAPI {
  private baseUrl: string
  private apiKey: string
  private apiSecret: string
  private authToken?: string

  constructor(baseUrl: string = 'http://localhost:8000/api/v1', apiKey?: string, apiSecret?: string) {
    this.baseUrl = baseUrl
    this.apiKey = apiKey || ''
    this.apiSecret = apiSecret || ''
  }

  /**
   * Set authentication credentials
   */
  setCredentials(apiKey: string, apiSecret: string): void {
    this.apiKey = apiKey
    this.apiSecret = apiSecret
  }

  /**
   * Authenticate with the trading system
   */
  async authenticate(): Promise<AuthResponse> {
    const timestamp = Date.now()
    const signature = this.generateSignature(timestamp)

    const authRequest: AuthRequest = {
      apiKey: this.apiKey,
      signature,
      timestamp
    }

    const response = await this.request<AuthResponse>('POST', '/auth/login', authRequest)
    
    if (response.success && response.data?.token) {
      this.authToken = response.data.token
    }

    return response.data!
  }

  // Order Management
  /**
   * Create a new order
   */
  async createOrder(orderRequest: CreateOrderRequest): Promise<Order> {
    const response = await this.request<Order>('POST', '/orders', orderRequest)
    return response.data!
  }

  /**
   * Create an algorithmic order
   */
  async createAlgorithmicOrder(orderRequest: AlgorithmicOrderRequest): Promise<Order> {
    const response = await this.request<Order>('POST', '/orders/algorithmic', orderRequest)
    return response.data!
  }

  /**
   * Get order by ID
   */
  async getOrder(orderId: string): Promise<Order> {
    const response = await this.request<Order>('GET', `/orders/${orderId}`)
    return response.data!
  }

  /**
   * Get all orders with optional filtering
   */
  async getOrders(filter?: OrderFilter, pagination?: PaginationParams): Promise<PaginatedResponse<Order>> {
    const params = new URLSearchParams()
    
    if (filter) {
      Object.entries(filter).forEach(([key, value]) => {
        if (value !== undefined) {
          params.append(key, String(value))
        }
      })
    }

    if (pagination) {
      Object.entries(pagination).forEach(([key, value]) => {
        if (value !== undefined) {
          params.append(key, String(value))
        }
      })
    }

    const response = await this.request<PaginatedResponse<Order>>('GET', `/orders?${params.toString()}`)
    return response.data!
  }

  /**
   * Modify an existing order
   */
  async modifyOrder(modifyRequest: ModifyOrderRequest): Promise<Order> {
    const { orderId, ...updates } = modifyRequest
    const response = await this.request<Order>('PUT', `/orders/${orderId}`, updates)
    return response.data!
  }

  /**
   * Cancel an order
   */
  async cancelOrder(orderId: string): Promise<void> {
    await this.request('DELETE', `/orders/${orderId}`)
  }

  /**
   * Cancel all orders for a symbol
   */
  async cancelAllOrders(symbol?: string): Promise<void> {
    const params = symbol ? `?symbol=${symbol}` : ''
    await this.request('DELETE', `/orders/all${params}`)
  }

  // Position Management
  /**
   * Get all positions
   */
  async getPositions(): Promise<Position[]> {
    const response = await this.request<Position[]>('GET', '/positions')
    return response.data!
  }

  /**
   * Get position for a specific symbol
   */
  async getPosition(symbol: string): Promise<Position> {
    const response = await this.request<Position>('GET', `/positions/${symbol}`)
    return response.data!
  }

  /**
   * Close a position
   */
  async closePosition(symbol: string, quantity?: number): Promise<void> {
    const body = quantity ? { quantity } : {}
    await this.request('POST', `/positions/${symbol}/close`, body)
  }

  // Portfolio Management
  /**
   * Get portfolio summary
   */
  async getPortfolio(): Promise<Portfolio> {
    const response = await this.request<Portfolio>('GET', '/portfolio')
    return response.data!
  }

  /**
   * Get account balances
   */
  async getBalances(): Promise<Balance[]> {
    const response = await this.request<Balance[]>('GET', '/balances')
    return response.data!
  }

  // Risk Management
  /**
   * Get risk metrics
   */
  async getRiskMetrics(): Promise<RiskMetrics> {
    const response = await this.request<RiskMetrics>('GET', '/risk/metrics')
    return response.data!
  }

  /**
   * Get risk limits
   */
  async getRiskLimits(): Promise<RiskLimits> {
    const response = await this.request<RiskLimits>('GET', '/risk/limits')
    return response.data!
  }

  /**
   * Update risk limits
   */
  async updateRiskLimits(limits: Partial<RiskLimits>): Promise<RiskLimits> {
    const response = await this.request<RiskLimits>('PUT', '/risk/limits', limits)
    return response.data!
  }

  // Performance Analytics
  /**
   * Get performance metrics
   */
  async getPerformanceMetrics(timeframe?: string): Promise<PerformanceMetrics> {
    const params = timeframe ? `?timeframe=${timeframe}` : ''
    const response = await this.request<PerformanceMetrics>('GET', `/performance/metrics${params}`)
    return response.data!
  }

  /**
   * Get trade history
   */
  async getTradeHistory(filter?: TradeFilter, pagination?: PaginationParams): Promise<PaginatedResponse<any>> {
    const params = new URLSearchParams()
    
    if (filter) {
      Object.entries(filter).forEach(([key, value]) => {
        if (value !== undefined) {
          params.append(key, String(value))
        }
      })
    }

    if (pagination) {
      Object.entries(pagination).forEach(([key, value]) => {
        if (value !== undefined) {
          params.append(key, String(value))
        }
      })
    }

    const response = await this.request<PaginatedResponse<any>>('GET', `/trades?${params.toString()}`)
    return response.data!
  }

  // System Monitoring
  /**
   * Get system health
   */
  async getSystemHealth(): Promise<SystemHealth> {
    const response = await this.request<SystemHealth>('GET', '/system/health')
    return response.data!
  }

  /**
   * Get system components status
   */
  async getSystemComponents(): Promise<SystemComponent[]> {
    const response = await this.request<SystemComponent[]>('GET', '/system/components')
    return response.data!
  }

  // Configuration
  /**
   * Get trading configuration
   */
  async getTradingConfig(): Promise<TradingConfig> {
    const response = await this.request<TradingConfig>('GET', '/config/trading')
    return response.data!
  }

  /**
   * Update trading configuration
   */
  async updateTradingConfig(config: Partial<TradingConfig>): Promise<TradingConfig> {
    const response = await this.request<TradingConfig>('PUT', '/config/trading', config)
    return response.data!
  }

  // Strategy Management
  /**
   * Get all trading strategies
   */
  async getStrategies(): Promise<TradingStrategy[]> {
    const response = await this.request<TradingStrategy[]>('GET', '/strategies')
    return response.data!
  }

  /**
   * Get strategy by ID
   */
  async getStrategy(strategyId: string): Promise<TradingStrategy> {
    const response = await this.request<TradingStrategy>('GET', `/strategies/${strategyId}`)
    return response.data!
  }

  /**
   * Create a new strategy
   */
  async createStrategy(strategy: Omit<TradingStrategy, 'id' | 'createdAt' | 'updatedAt'>): Promise<TradingStrategy> {
    const response = await this.request<TradingStrategy>('POST', '/strategies', strategy)
    return response.data!
  }

  /**
   * Update a strategy
   */
  async updateStrategy(strategyId: string, updates: Partial<TradingStrategy>): Promise<TradingStrategy> {
    const response = await this.request<TradingStrategy>('PUT', `/strategies/${strategyId}`, updates)
    return response.data!
  }

  /**
   * Delete a strategy
   */
  async deleteStrategy(strategyId: string): Promise<void> {
    await this.request('DELETE', `/strategies/${strategyId}`)
  }

  // Private helper methods
  private async request<T = any>(
    method: string,
    endpoint: string,
    body?: any
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    }

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`
    }

    const config: RequestInit = {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined
    }

    try {
      const response = await fetch(url, config)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}: ${response.statusText}`)
      }

      return data
    } catch (error) {
      console.error(`API request failed: ${method} ${endpoint}`, error)
      throw error
    }
  }

  private generateSignature(timestamp: number): string {
    // This should implement proper HMAC-SHA256 signature generation
    // For now, return a mock signature
    const message = `${timestamp}${this.apiKey}`
    return btoa(message) // Base64 encode as mock signature
  }
}

// Singleton instance
export const tradingAPI = new TradingAPI()
