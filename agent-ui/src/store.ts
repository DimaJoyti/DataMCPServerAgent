import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

import {
  type PlaygroundChatMessage,
  type SessionEntry
} from '@/types/playground'
import {
  type InfiniteLoopConfig,
  type InfiniteLoopSession,
  type InfiniteLoopUIState,
  type InfiniteLoopError,
  type WebSocketMessage
} from '@/types/infinite-loop'

interface Agent {
  value: string
  label: string
  model: {
    provider: string
  }
  storage?: boolean
}

interface PlaygroundStore {
  hydrated: boolean
  setHydrated: () => void
  streamingErrorMessage: string
  setStreamingErrorMessage: (streamingErrorMessage: string) => void
  endpoints: {
    endpoint: string
    id_playground_endpoint: string
  }[]
  setEndpoints: (
    endpoints: {
      endpoint: string
      id_playground_endpoint: string
    }[]
  ) => void
  isStreaming: boolean
  setIsStreaming: (isStreaming: boolean) => void
  isEndpointActive: boolean
  setIsEndpointActive: (isActive: boolean) => void
  isEndpointLoading: boolean
  setIsEndpointLoading: (isLoading: boolean) => void
  messages: PlaygroundChatMessage[]
  setMessages: (
    messages:
      | PlaygroundChatMessage[]
      | ((prevMessages: PlaygroundChatMessage[]) => PlaygroundChatMessage[])
  ) => void
  hasStorage: boolean
  setHasStorage: (hasStorage: boolean) => void
  chatInputRef: React.RefObject<HTMLTextAreaElement | null>
  selectedEndpoint: string
  setSelectedEndpoint: (selectedEndpoint: string) => void
  agents: Agent[]
  setAgents: (agents: Agent[]) => void
  selectedModel: string
  setSelectedModel: (model: string) => void
  sessionsData: SessionEntry[] | null
  setSessionsData: (
    sessionsData:
      | SessionEntry[]
      | ((prevSessions: SessionEntry[] | null) => SessionEntry[] | null)
  ) => void
  isSessionsLoading: boolean
  setIsSessionsLoading: (isSessionsLoading: boolean) => void
}

interface InfiniteLoopStore {
  // Core state
  currentSession: InfiniteLoopSession | null
  setCurrentSession: (session: InfiniteLoopSession | null) => void

  // UI state
  uiState: InfiniteLoopUIState
  setUIState: (state: Partial<InfiniteLoopUIState>) => void

  // Configuration
  defaultConfig: InfiniteLoopConfig
  setDefaultConfig: (config: InfiniteLoopConfig) => void

  // Sessions management
  sessions: InfiniteLoopSession[]
  setSessions: (sessions: InfiniteLoopSession[]) => void
  addSession: (session: InfiniteLoopSession) => void
  updateSession: (sessionId: string, updates: Partial<InfiniteLoopSession>) => void
  removeSession: (sessionId: string) => void

  // WebSocket connection
  wsConnection: WebSocket | null
  setWSConnection: (ws: WebSocket | null) => void
  isConnected: boolean
  setIsConnected: (connected: boolean) => void

  // Real-time updates
  lastMessage: WebSocketMessage | null
  setLastMessage: (message: WebSocketMessage | null) => void

  // Error handling
  errors: InfiniteLoopError[]
  addError: (error: InfiniteLoopError) => void
  clearErrors: () => void
  removeError: (index: number) => void
}

export const usePlaygroundStore = create<PlaygroundStore>()(
  persist(
    (set) => ({
      hydrated: false,
      setHydrated: () => set({ hydrated: true }),
      streamingErrorMessage: '',
      setStreamingErrorMessage: (streamingErrorMessage) =>
        set(() => ({ streamingErrorMessage })),
      endpoints: [],
      setEndpoints: (endpoints) => set(() => ({ endpoints })),
      isStreaming: false,
      setIsStreaming: (isStreaming) => set(() => ({ isStreaming })),
      isEndpointActive: false,
      setIsEndpointActive: (isActive) =>
        set(() => ({ isEndpointActive: isActive })),
      isEndpointLoading: true,
      setIsEndpointLoading: (isLoading) =>
        set(() => ({ isEndpointLoading: isLoading })),
      messages: [],
      setMessages: (messages) =>
        set((state) => ({
          messages:
            typeof messages === 'function' ? messages(state.messages) : messages
        })),
      hasStorage: false,
      setHasStorage: (hasStorage) => set(() => ({ hasStorage })),
      chatInputRef: { current: null },
      selectedEndpoint: 'http://localhost:8001',
      setSelectedEndpoint: (selectedEndpoint) =>
        set(() => ({ selectedEndpoint })),
      agents: [],
      setAgents: (agents) => set({ agents }),
      selectedModel: '',
      setSelectedModel: (selectedModel) => set(() => ({ selectedModel })),
      sessionsData: null,
      setSessionsData: (sessionsData) =>
        set((state) => ({
          sessionsData:
            typeof sessionsData === 'function'
              ? sessionsData(state.sessionsData)
              : sessionsData
        })),
      isSessionsLoading: false,
      setIsSessionsLoading: (isSessionsLoading) =>
        set(() => ({ isSessionsLoading }))
    }),
    {
      name: 'endpoint-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        selectedEndpoint: state.selectedEndpoint
      }),
      onRehydrateStorage: () => (state) => {
        state?.setHydrated?.()
      }
    }
  )
)

// Default configuration for infinite loop
const defaultInfiniteLoopConfig: InfiniteLoopConfig = {
  maxParallelAgents: 5,
  waveSizeMin: 3,
  waveSizeMax: 5,
  contextThreshold: 0.8,
  qualityThreshold: 0.7,
  uniquenessThreshold: 0.8,
  validationEnabled: true,
  maxRetries: 3,
  retryDelay: 1.0,
  errorRecoveryEnabled: true,
  batchProcessing: true,
  asyncExecution: true,
  memoryOptimization: true,
  logLevel: 'INFO',
  detailedLogging: false
}

// Default UI state
const defaultUIState: InfiniteLoopUIState = {
  isConnected: false,
  isExecuting: false,
  selectedConfig: defaultInfiniteLoopConfig,
  realtimeUpdates: true,
  selectedView: 'dashboard',
  errors: []
}

export const useInfiniteLoopStore = create<InfiniteLoopStore>()((set, get) => ({
  // Core state
  currentSession: null,
  setCurrentSession: (session) => set({ currentSession: session }),

  // UI state
  uiState: defaultUIState,
  setUIState: (state) => set((prev) => ({
    uiState: { ...prev.uiState, ...state }
  })),

  // Configuration
  defaultConfig: defaultInfiniteLoopConfig,
  setDefaultConfig: (config) => set({ defaultConfig: config }),

  // Sessions management
  sessions: [],
  setSessions: (sessions) => set({ sessions }),
  addSession: (session) => set((state) => ({
    sessions: [...state.sessions, session]
  })),
  updateSession: (sessionId, updates) => set((state) => ({
    sessions: state.sessions.map(session =>
      session.sessionId === sessionId
        ? { ...session, ...updates, updatedAt: new Date().toISOString() }
        : session
    ),
    currentSession: state.currentSession?.sessionId === sessionId
      ? { ...state.currentSession, ...updates, updatedAt: new Date().toISOString() }
      : state.currentSession
  })),
  removeSession: (sessionId) => set((state) => ({
    sessions: state.sessions.filter(session => session.sessionId !== sessionId),
    currentSession: state.currentSession?.sessionId === sessionId
      ? null
      : state.currentSession
  })),

  // WebSocket connection
  wsConnection: null,
  setWSConnection: (ws) => set({ wsConnection: ws }),
  isConnected: false,
  setIsConnected: (connected) => set({ isConnected: connected }),

  // Real-time updates
  lastMessage: null,
  setLastMessage: (message) => set({ lastMessage: message }),

  // Error handling
  errors: [],
  addError: (error) => set((state) => ({
    errors: [...state.errors, error]
  })),
  clearErrors: () => set({ errors: [] }),
  removeError: (index) => set((state) => ({
    errors: state.errors.filter((_, i) => i !== index)
  }))
}))
