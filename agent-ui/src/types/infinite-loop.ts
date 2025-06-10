/**
 * TypeScript interfaces for the Infinite Agentic Loop system
 * These interfaces match the backend Python data structures
 */

export interface InfiniteLoopConfig {
  // Core settings
  maxParallelAgents: number
  waveSizeMin: number
  waveSizeMax: number
  contextThreshold: number
  maxIterations?: number

  // Quality control
  qualityThreshold: number
  uniquenessThreshold: number
  validationEnabled: boolean

  // Error handling
  maxRetries: number
  retryDelay: number
  errorRecoveryEnabled: boolean

  // Performance
  batchProcessing: boolean
  asyncExecution: boolean
  memoryOptimization: boolean

  // Logging
  logLevel: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
  detailedLogging: boolean
}

export interface ExecutionState {
  // Execution metadata
  sessionId: string
  startTime: string
  currentWave: number
  totalIterations: number

  // Status tracking
  isRunning: boolean
  isInfinite: boolean
  contextUsage: number

  // Results
  completedIterations: string[]
  failedIterations: string[]
  activeAgents: Record<string, string>

  // Performance metrics
  averageIterationTime: number
  successRate: number
  qualityScore: number
}

export interface SpecificationAnalysis {
  contentType: string
  format: string
  evolutionPattern: string
  innovationAreas: string[]
  qualityRequirements: Record<string, any>
  requirements: string[]
  constraints: string[]
}

export interface DirectoryState {
  existingFiles: string[]
  highestIteration: number
  evolutionSummary: string
  patterns: string[]
  totalSize: number
  lastModified: string
}

export interface IterationStrategy {
  startingIteration: number
  targetCount: number | 'infinite'
  isInfinite: boolean
  waveStrategy: WaveStrategy
  innovationDimensions: string[]
  qualityRequirements: Record<string, any>
}

export interface WaveStrategy {
  type: 'single_wave' | 'batched_waves' | 'large_batched_waves' | 'infinite_waves'
  waveSize: number
  maxWaves?: number
  contextMonitoring: boolean
}

export interface WaveResult {
  waveNumber: number
  startTime: string
  endTime: string
  completedIterations: number
  failedIterations: number
  totalTime: number
  averageIterationTime: number
  successRate: number
  qualityScore: number
  tasks: TaskResult[]
  errors: string[]
}

export interface TaskResult {
  iterationNumber: number
  agentId: string
  innovationDimension: string
  startTime: string
  endTime: string
  success: boolean
  outputPath?: string
  qualityScore?: number
  errorMessage?: string
  metrics: TaskMetrics
}

export interface TaskMetrics {
  executionTime: number
  memoryUsage: number
  contextTokens: number
  outputSize: number
  validationScore: number
}

export interface AgentStatus {
  agentId: string
  status: 'idle' | 'running' | 'completed' | 'error'
  currentTask?: string
  iterationNumber?: number
  startTime?: string
  progress: number
  metrics: AgentMetrics
}

export interface AgentMetrics {
  totalTasks: number
  completedTasks: number
  failedTasks: number
  averageExecutionTime: number
  successRate: number
  qualityScore: number
}

export interface QualityMetrics {
  overallScore: number
  uniquenessScore: number
  validationScore: number
  consistencyScore: number
  innovationScore: number
  trends: QualityTrend[]
}

export interface QualityTrend {
  iteration: number
  score: number
  timestamp: string
}

export interface PerformanceMetrics {
  executionTimeSeconds: number
  totalIterations: number
  completedIterations: number
  failedIterations: number
  successRate: number
  averageIterationTime: number
  qualityScore: number
  wavesCompleted: number
  contextUsage: number
  memoryUsage: number
  throughput: number
}

export interface InfiniteLoopSession {
  sessionId: string
  config: InfiniteLoopConfig
  executionState: ExecutionState
  specAnalysis?: SpecificationAnalysis
  directoryState?: DirectoryState
  iterationStrategy?: IterationStrategy
  waves: WaveResult[]
  agents: AgentStatus[]
  qualityMetrics: QualityMetrics
  performanceMetrics: PerformanceMetrics
  createdAt: string
  updatedAt: string
}

export interface InfiniteLoopExecutionRequest {
  specFile: File | string
  outputDir: string
  count: number | 'infinite'
  config?: Partial<InfiniteLoopConfig>
}

export interface InfiniteLoopExecutionResponse {
  success: boolean
  sessionId: string
  executionState: ExecutionState
  results?: any
  statistics?: PerformanceMetrics
  error?: string
}

export interface WebSocketMessage {
  type: 'execution_started' | 'wave_started' | 'wave_completed' | 'task_completed' | 'execution_completed' | 'error' | 'metrics_update'
  sessionId: string
  timestamp: string
  data: any
}

export interface InfiniteLoopError {
  code: string
  message: string
  details?: any
  timestamp: string
  sessionId?: string
  waveNumber?: number
  iterationNumber?: number
}

// UI-specific interfaces
export interface InfiniteLoopUIState {
  currentSession?: InfiniteLoopSession
  isConnected: boolean
  isExecuting: boolean
  selectedConfig: InfiniteLoopConfig
  realtimeUpdates: boolean
  selectedView: 'dashboard' | 'configuration' | 'monitoring' | 'analytics'
  errors: InfiniteLoopError[]
}

export type InfiniteLoopMode = 'setup' | 'executing' | 'completed' | 'error'

export interface ChartDataPoint {
  x: number | string
  y: number
  label?: string
  color?: string
}

export interface TimeSeriesData {
  timestamp: string
  value: number
  label?: string
}
