/**
 * API Types and Interfaces
 * Shared types for API communication
 */

// Base API Response
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
  timestamp: number
}

// Order Types
export interface CreateOrderRequest {
  symbol: string
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stop' | 'stop-limit'
  quantity: number
  price?: number
  stopPrice?: number
  timeInForce?: 'day' | 'gtc' | 'ioc' | 'fok'
  clientOrderId?: string
}

export interface Order {
  orderId: string
  clientOrderId?: string
  symbol: string
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stop' | 'stop-limit'
  status: 'pending' | 'filled' | 'partially_filled' | 'cancelled' | 'rejected'
  quantity: number
  price?: number
  stopPrice?: number
  filledQuantity: number
  averageFillPrice?: number
  commission: number
  timeInForce: 'day' | 'gtc' | 'ioc' | 'fok'
  createdAt: string
  updatedAt: string
}

export interface ModifyOrderRequest {
  orderId: string
  quantity?: number
  price?: number
  stopPrice?: number
}

// Position Types
export interface Position {
  symbol: string
  side: 'long' | 'short'
  quantity: number
  entryPrice: number
  currentPrice: number
  unrealizedPnL: number
  realizedPnL: number
  totalValue: number
  stopLoss?: number
  takeProfit?: number
  riskLevel: 'low' | 'medium' | 'high'
  openDate: string
}

// Portfolio Types
export interface Portfolio {
  totalValue: number
  totalPnL: number
  totalPnLPercent: number
  dayPnL: number
  dayPnLPercent: number
  marginUsed: number
  marginAvailable: number
  buyingPower: number
  positions: Position[]
}

// Balance Types
export interface Balance {
  currency: string
  available: number
  locked: number
  total: number
}

// Market Data Types
export interface MarketData {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  high24h: number
  low24h: number
  bid: number
  ask: number
  timestamp: number
}

export interface OrderBookData {
  symbol: string
  bids: Array<{
    price: number
    size: number
    orders: number
  }>
  asks: Array<{
    price: number
    size: number
    orders: number
  }>
  timestamp: number
}

// Risk Types
export interface RiskMetrics {
  valueAtRisk: number
  maxDrawdown: number
  sharpeRatio: number
  beta: number
  alpha: number
  volatility: number
  concentrationRisk: number
  leverageRatio: number
}

export interface RiskLimits {
  maxPositionSize: number
  maxDailyLoss: number
  maxLeverage: number
  maxConcentration: number
  stopLossRequired: boolean
}

// Performance Types
export interface PerformanceMetrics {
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
  winRate: number
  profitFactor: number
  avgWin: number
  avgLoss: number
  totalTrades: number
  winningTrades: number
  losingTrades: number
}

// System Types
export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'down'
  uptime: number
  latency: number
  throughput: number
  errorRate: number
  lastUpdate: string
}

export interface SystemComponent {
  name: string
  status: 'online' | 'degraded' | 'offline'
  uptime: number
  latency?: number
  throughput?: number
  errorRate?: number
  lastUpdate: string
}

// Authentication Types
export interface AuthRequest {
  apiKey: string
  signature: string
  timestamp: number
}

export interface AuthResponse {
  success: boolean
  token?: string
  expiresAt?: number
  permissions?: string[]
  error?: string
}

// Pagination
export interface PaginationParams {
  page?: number
  limit?: number
  sortBy?: string
  sortOrder?: 'asc' | 'desc'
}

export interface PaginatedResponse<T> {
  data: T[]
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
    hasNext: boolean
    hasPrev: boolean
  }
}

// Filter Types
export interface OrderFilter {
  symbol?: string
  side?: 'buy' | 'sell'
  status?: string
  startDate?: string
  endDate?: string
}

export interface TradeFilter {
  symbol?: string
  side?: 'buy' | 'sell'
  startDate?: string
  endDate?: string
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: string
  data?: any
  timestamp: number
  id?: string
}

// Error Types
export interface ApiError {
  code: string
  message: string
  details?: any
  timestamp: number
}

// Configuration Types
export interface TradingConfig {
  riskLimits: RiskLimits
  defaultTimeInForce: 'day' | 'gtc' | 'ioc' | 'fok'
  enableAlgorithmicTrading: boolean
  enableRiskManagement: boolean
  maxOrdersPerSecond: number
}

// Strategy Types
export interface TradingStrategy {
  id: string
  name: string
  description: string
  status: 'active' | 'inactive' | 'paused'
  parameters: Record<string, any>
  performance: {
    totalReturn: number
    sharpeRatio: number
    maxDrawdown: number
    winRate: number
  }
  createdAt: string
  updatedAt: string
}

// Execution Algorithm Types
export interface ExecutionAlgorithm {
  type: 'twap' | 'vwap' | 'iceberg' | 'implementation_shortfall'
  parameters: Record<string, any>
}

export interface AlgorithmicOrderRequest extends CreateOrderRequest {
  algorithm: ExecutionAlgorithm
}

// Historical Data Types
export interface HistoricalDataRequest {
  symbol: string
  interval: '1m' | '5m' | '15m' | '1h' | '4h' | '1d'
  startTime: number
  endTime: number
  limit?: number
}

export interface OHLCV {
  timestamp: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

// News and Events
export interface NewsItem {
  id: string
  title: string
  content: string
  source: string
  symbols: string[]
  sentiment: 'positive' | 'negative' | 'neutral'
  publishedAt: string
}

export interface EconomicEvent {
  id: string
  name: string
  country: string
  currency: string
  impact: 'low' | 'medium' | 'high'
  actual?: number
  forecast?: number
  previous?: number
  scheduledAt: string
}
