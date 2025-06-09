/**
 * Market Data API Service
 * Handles REST API calls for market data operations
 */

import {
  ApiResponse,
  MarketData,
  OrderBookData,
  HistoricalDataRequest,
  OHLCV,
  NewsItem,
  EconomicEvent,
  PaginationParams,
  PaginatedResponse
} from './types'

export class MarketDataAPI {
  private baseUrl: string
  private apiKey?: string

  constructor(baseUrl: string = 'http://localhost:8000/api/v1', apiKey?: string) {
    this.baseUrl = baseUrl
    this.apiKey = apiKey
  }

  /**
   * Set API key for authentication
   */
  setApiKey(apiKey: string): void {
    this.apiKey = apiKey
  }

  // Real-time Market Data
  /**
   * Get current market data for a symbol
   */
  async getMarketData(symbol: string): Promise<MarketData> {
    const response = await this.request<MarketData>('GET', `/market-data/${symbol}`)
    return response.data!
  }

  /**
   * Get market data for multiple symbols
   */
  async getMultipleMarketData(symbols: string[]): Promise<MarketData[]> {
    const symbolsParam = symbols.join(',')
    const response = await this.request<MarketData[]>('GET', `/market-data?symbols=${symbolsParam}`)
    return response.data!
  }

  /**
   * Get order book for a symbol
   */
  async getOrderBook(symbol: string, depth: number = 20): Promise<OrderBookData> {
    const response = await this.request<OrderBookData>('GET', `/market-data/${symbol}/orderbook?depth=${depth}`)
    return response.data!
  }

  /**
   * Get recent trades for a symbol
   */
  async getRecentTrades(symbol: string, limit: number = 100): Promise<any[]> {
    const response = await this.request<any[]>('GET', `/market-data/${symbol}/trades?limit=${limit}`)
    return response.data!
  }

  /**
   * Get 24hr ticker statistics
   */
  async getTicker24hr(symbol?: string): Promise<any> {
    const endpoint = symbol ? `/market-data/${symbol}/ticker` : '/market-data/ticker'
    const response = await this.request<any>('GET', endpoint)
    return response.data!
  }

  // Historical Data
  /**
   * Get historical OHLCV data
   */
  async getHistoricalData(request: HistoricalDataRequest): Promise<OHLCV[]> {
    const params = new URLSearchParams({
      interval: request.interval,
      startTime: request.startTime.toString(),
      endTime: request.endTime.toString()
    })

    if (request.limit) {
      params.append('limit', request.limit.toString())
    }

    const response = await this.request<OHLCV[]>('GET', `/market-data/${request.symbol}/history?${params.toString()}`)
    return response.data!
  }

  /**
   * Get historical trades
   */
  async getHistoricalTrades(
    symbol: string,
    startTime: number,
    endTime: number,
    limit?: number
  ): Promise<any[]> {
    const params = new URLSearchParams({
      startTime: startTime.toString(),
      endTime: endTime.toString()
    })

    if (limit) {
      params.append('limit', limit.toString())
    }

    const response = await this.request<any[]>('GET', `/market-data/${symbol}/trades/history?${params.toString()}`)
    return response.data!
  }

  // Market Statistics
  /**
   * Get market statistics
   */
  async getMarketStats(): Promise<any> {
    const response = await this.request<any>('GET', '/market-data/stats')
    return response.data!
  }

  /**
   * Get top gainers/losers
   */
  async getTopMovers(type: 'gainers' | 'losers' = 'gainers', limit: number = 10): Promise<any[]> {
    const response = await this.request<any[]>('GET', `/market-data/movers/${type}?limit=${limit}`)
    return response.data!
  }

  /**
   * Get most active symbols by volume
   */
  async getMostActive(limit: number = 10): Promise<any[]> {
    const response = await this.request<any[]>('GET', `/market-data/active?limit=${limit}`)
    return response.data!
  }

  // Symbol Information
  /**
   * Get all available symbols
   */
  async getSymbols(): Promise<any[]> {
    const response = await this.request<any[]>('GET', '/market-data/symbols')
    return response.data!
  }

  /**
   * Get symbol information
   */
  async getSymbolInfo(symbol: string): Promise<any> {
    const response = await this.request<any>('GET', `/market-data/symbols/${symbol}`)
    return response.data!
  }

  /**
   * Search symbols
   */
  async searchSymbols(query: string, limit: number = 20): Promise<any[]> {
    const response = await this.request<any[]>('GET', `/market-data/symbols/search?q=${query}&limit=${limit}`)
    return response.data!
  }

  // News and Events
  /**
   * Get market news
   */
  async getNews(
    symbols?: string[],
    pagination?: PaginationParams
  ): Promise<PaginatedResponse<NewsItem>> {
    const params = new URLSearchParams()

    if (symbols && symbols.length > 0) {
      params.append('symbols', symbols.join(','))
    }

    if (pagination) {
      Object.entries(pagination).forEach(([key, value]) => {
        if (value !== undefined) {
          params.append(key, String(value))
        }
      })
    }

    const response = await this.request<PaginatedResponse<NewsItem>>('GET', `/market-data/news?${params.toString()}`)
    return response.data!
  }

  /**
   * Get economic calendar events
   */
  async getEconomicEvents(
    startDate?: string,
    endDate?: string,
    impact?: 'low' | 'medium' | 'high'
  ): Promise<EconomicEvent[]> {
    const params = new URLSearchParams()

    if (startDate) params.append('startDate', startDate)
    if (endDate) params.append('endDate', endDate)
    if (impact) params.append('impact', impact)

    const response = await this.request<EconomicEvent[]>('GET', `/market-data/events?${params.toString()}`)
    return response.data!
  }

  // Market Hours
  /**
   * Get market hours for exchanges
   */
  async getMarketHours(exchange?: string): Promise<any> {
    const endpoint = exchange ? `/market-data/hours/${exchange}` : '/market-data/hours'
    const response = await this.request<any>('GET', endpoint)
    return response.data!
  }

  /**
   * Check if market is open
   */
  async isMarketOpen(exchange?: string): Promise<boolean> {
    const endpoint = exchange ? `/market-data/hours/${exchange}/status` : '/market-data/hours/status'
    const response = await this.request<{ isOpen: boolean }>('GET', endpoint)
    return response.data!.isOpen
  }

  // Technical Indicators
  /**
   * Get technical indicators for a symbol
   */
  async getTechnicalIndicators(
    symbol: string,
    indicators: string[],
    interval: string = '1d',
    period: number = 14
  ): Promise<any> {
    const params = new URLSearchParams({
      indicators: indicators.join(','),
      interval,
      period: period.toString()
    })

    const response = await this.request<any>('GET', `/market-data/${symbol}/indicators?${params.toString()}`)
    return response.data!
  }

  // Cryptocurrency specific (if applicable)
  /**
   * Get cryptocurrency market cap data
   */
  async getCryptoMarketCap(limit: number = 100): Promise<any[]> {
    const response = await this.request<any[]>('GET', `/market-data/crypto/marketcap?limit=${limit}`)
    return response.data!
  }

  /**
   * Get cryptocurrency fear & greed index
   */
  async getFearGreedIndex(): Promise<any> {
    const response = await this.request<any>('GET', '/market-data/crypto/fear-greed')
    return response.data!
  }

  // Forex specific (if applicable)
  /**
   * Get currency exchange rates
   */
  async getExchangeRates(baseCurrency: string = 'USD'): Promise<any> {
    const response = await this.request<any>('GET', `/market-data/forex/rates?base=${baseCurrency}`)
    return response.data!
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

    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey
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
      console.error(`Market Data API request failed: ${method} ${endpoint}`, error)
      throw error
    }
  }
}

// Singleton instance
export const marketDataAPI = new MarketDataAPI()

// Factory for different data providers
export class MarketDataAPIFactory {
  static createAlpacaAPI(apiKey: string): MarketDataAPI {
    return new MarketDataAPI('https://data.alpaca.markets/v2', apiKey)
  }

  static createPolygonAPI(apiKey: string): MarketDataAPI {
    return new MarketDataAPI('https://api.polygon.io/v2', apiKey)
  }

  static createBinanceAPI(): MarketDataAPI {
    return new MarketDataAPI('https://api.binance.com/api/v3')
  }

  static createLocalAPI(port: number = 8000): MarketDataAPI {
    return new MarketDataAPI(`http://localhost:${port}/api/v1`)
  }
}
