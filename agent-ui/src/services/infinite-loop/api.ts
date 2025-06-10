/**
 * API service for Infinite Agentic Loop backend communication
 */

import {
  type InfiniteLoopConfig,
  type InfiniteLoopSession,
  type InfiniteLoopExecutionRequest,
  type InfiniteLoopExecutionResponse,
  type PerformanceMetrics,
  type ExecutionState
} from '@/types/infinite-loop'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

class InfiniteLoopAPI {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  /**
   * Start infinite loop execution
   */
  async startExecution(request: InfiniteLoopExecutionRequest): Promise<InfiniteLoopExecutionResponse> {
    const formData = new FormData()
    
    if (request.specFile instanceof File) {
      formData.append('spec_file', request.specFile)
    } else {
      formData.append('spec_file_path', request.specFile)
    }
    
    formData.append('output_dir', request.outputDir)
    formData.append('count', request.count.toString())
    
    if (request.config) {
      formData.append('config', JSON.stringify(request.config))
    }

    const response = await fetch(`${this.baseUrl}/infinite-loop/execute`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`Failed to start execution: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get execution status
   */
  async getExecutionStatus(sessionId: string): Promise<ExecutionState> {
    const response = await fetch(`${this.baseUrl}/infinite-loop/status/${sessionId}`)
    
    if (!response.ok) {
      throw new Error(`Failed to get status: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Stop execution
   */
  async stopExecution(sessionId: string): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${this.baseUrl}/infinite-loop/stop/${sessionId}`, {
      method: 'POST'
    })

    if (!response.ok) {
      throw new Error(`Failed to stop execution: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Pause execution
   */
  async pauseExecution(sessionId: string): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${this.baseUrl}/infinite-loop/pause/${sessionId}`, {
      method: 'POST'
    })

    if (!response.ok) {
      throw new Error(`Failed to pause execution: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Resume execution
   */
  async resumeExecution(sessionId: string): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${this.baseUrl}/infinite-loop/resume/${sessionId}`, {
      method: 'POST'
    })

    if (!response.ok) {
      throw new Error(`Failed to resume execution: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get performance metrics
   */
  async getMetrics(sessionId: string): Promise<PerformanceMetrics> {
    const response = await fetch(`${this.baseUrl}/infinite-loop/metrics/${sessionId}`)
    
    if (!response.ok) {
      throw new Error(`Failed to get metrics: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get session details
   */
  async getSession(sessionId: string): Promise<InfiniteLoopSession> {
    const response = await fetch(`${this.baseUrl}/infinite-loop/session/${sessionId}`)
    
    if (!response.ok) {
      throw new Error(`Failed to get session: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List all sessions
   */
  async listSessions(): Promise<InfiniteLoopSession[]> {
    const response = await fetch(`${this.baseUrl}/infinite-loop/sessions`)
    
    if (!response.ok) {
      throw new Error(`Failed to list sessions: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Delete session
   */
  async deleteSession(sessionId: string): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${this.baseUrl}/infinite-loop/session/${sessionId}`, {
      method: 'DELETE'
    })

    if (!response.ok) {
      throw new Error(`Failed to delete session: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Update configuration
   */
  async updateConfig(sessionId: string, config: Partial<InfiniteLoopConfig>): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${this.baseUrl}/infinite-loop/config/${sessionId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(config)
    })

    if (!response.ok) {
      throw new Error(`Failed to update config: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get WebSocket URL for real-time updates
   */
  getWebSocketUrl(sessionId: string): string {
    const wsProtocol = this.baseUrl.startsWith('https') ? 'wss' : 'ws'
    const wsBaseUrl = this.baseUrl.replace(/^https?/, wsProtocol)
    return `${wsBaseUrl}/infinite-loop/stream/${sessionId}`
  }

  /**
   * Validate specification file
   */
  async validateSpecification(file: File): Promise<{ valid: boolean; errors: string[]; analysis?: any }> {
    const formData = new FormData()
    formData.append('spec_file', file)

    const response = await fetch(`${this.baseUrl}/infinite-loop/validate-spec`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`Failed to validate specification: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get available output directories
   */
  async getOutputDirectories(): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/infinite-loop/output-dirs`)
    
    if (!response.ok) {
      throw new Error(`Failed to get output directories: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Create new output directory
   */
  async createOutputDirectory(path: string): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${this.baseUrl}/infinite-loop/output-dirs`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ path })
    })

    if (!response.ok) {
      throw new Error(`Failed to create output directory: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get system health status
   */
  async getHealthStatus(): Promise<{ status: string; details: any }> {
    const response = await fetch(`${this.baseUrl}/infinite-loop/health`)
    
    if (!response.ok) {
      throw new Error(`Failed to get health status: ${response.statusText}`)
    }

    return response.json()
  }
}

// Export singleton instance
export const infiniteLoopAPI = new InfiniteLoopAPI()
export default infiniteLoopAPI
