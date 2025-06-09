/**
 * React hooks for Brand Agent functionality
 */

import { useState, useEffect, useCallback } from 'react'
import { 
  brandAgentAPI, 
  BrandAgent, 
  CreateBrandAgentRequest, 
  BrandSummary,
  KnowledgeItem,
  ConversationSession
} from '@/lib/brand-agent-api'

// Hook for managing brand agents
export const useBrandAgents = (brandId?: string) => {
  const [agents, setAgents] = useState<BrandAgent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchAgents = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      
      const filters = brandId ? { brandId } : undefined
      const data = await brandAgentAPI.getBrandAgents(filters)
      setAgents(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch agents')
    } finally {
      setLoading(false)
    }
  }, [brandId])

  useEffect(() => {
    fetchAgents()
  }, [fetchAgents])

  const createAgent = useCallback(async (data: CreateBrandAgentRequest) => {
    try {
      const newAgent = await brandAgentAPI.createBrandAgent(data)
      setAgents(prev => [...prev, newAgent])
      return newAgent
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create agent')
      throw err
    }
  }, [])

  const updateAgentPersonality = useCallback(async (
    agentId: string, 
    personality: BrandAgent['personality']
  ) => {
    try {
      const updatedAgent = await brandAgentAPI.updateAgentPersonality(agentId, personality)
      setAgents(prev => prev.map(agent => 
        agent.id === agentId ? updatedAgent : agent
      ))
      return updatedAgent
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update agent personality')
      throw err
    }
  }, [])

  const deployAgent = useCallback(async (agentId: string, channel: string) => {
    try {
      const updatedAgent = await brandAgentAPI.deployAgentToChannel(agentId, channel)
      setAgents(prev => prev.map(agent => 
        agent.id === agentId ? updatedAgent : agent
      ))
      return updatedAgent
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to deploy agent')
      throw err
    }
  }, [])

  return {
    agents,
    loading,
    error,
    refetch: fetchAgents,
    createAgent,
    updateAgentPersonality,
    deployAgent,
  }
}

// Hook for brand summary
export const useBrandSummary = (brandId: string) => {
  const [summary, setSummary] = useState<BrandSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchSummary = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      
      const data = await brandAgentAPI.getBrandSummary(brandId)
      setSummary(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch brand summary')
    } finally {
      setLoading(false)
    }
  }, [brandId])

  useEffect(() => {
    if (brandId) {
      fetchSummary()
    }
  }, [fetchSummary, brandId])

  return {
    summary,
    loading,
    error,
    refetch: fetchSummary,
  }
}

// Hook for agent performance
export const useAgentPerformance = (agentId?: string) => {
  const [performance, setPerformance] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchPerformance = useCallback(async (id: string) => {
    try {
      setLoading(true)
      setError(null)
      
      const data = await brandAgentAPI.getAgentPerformance(id)
      setPerformance(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch agent performance')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (agentId) {
      fetchPerformance(agentId)
    }
  }, [fetchPerformance, agentId])

  return {
    performance,
    loading,
    error,
    fetchPerformance,
  }
}

// Hook for knowledge management
export const useKnowledge = (brandId: string) => {
  const [knowledge, setKnowledge] = useState<KnowledgeItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const searchKnowledge = useCallback(async (
    query: string, 
    knowledgeType?: string
  ) => {
    try {
      setLoading(true)
      setError(null)
      
      const data = await brandAgentAPI.searchKnowledge(brandId, query, knowledgeType)
      setKnowledge(data)
      return data
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to search knowledge')
      throw err
    } finally {
      setLoading(false)
    }
  }, [brandId])

  const createKnowledgeItem = useCallback(async (data: {
    title: string
    content: string
    knowledgeType: string
    tags?: string[]
    priority?: number
    sourceUrl?: string
  }) => {
    try {
      const newItem = await brandAgentAPI.createKnowledgeItem({
        ...data,
        brandId,
      })
      setKnowledge(prev => [...prev, newItem])
      return newItem
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create knowledge item')
      throw err
    }
  }, [brandId])

  return {
    knowledge,
    loading,
    error,
    searchKnowledge,
    createKnowledgeItem,
  }
}

// Hook for conversations
export const useConversations = () => {
  const [sessions, setSessions] = useState<ConversationSession[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const startConversation = useCallback(async (data: {
    agentId: string
    channel: string
    userId?: string
  }) => {
    try {
      setLoading(true)
      setError(null)
      
      const session = await brandAgentAPI.startConversation(data)
      setSessions(prev => [...prev, session])
      return session
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start conversation')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const addMessage = useCallback(async (
    sessionId: string,
    data: {
      senderType: 'user' | 'agent'
      content: string
      messageType?: string
      metadata?: Record<string, any>
    }
  ) => {
    try {
      const message = await brandAgentAPI.addMessageToConversation(sessionId, data)
      return message
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add message')
      throw err
    }
  }, [])

  const endConversation = useCallback(async (
    sessionId: string,
    satisfactionRating?: number
  ) => {
    try {
      const session = await brandAgentAPI.endConversation(sessionId, satisfactionRating)
      setSessions(prev => prev.map(s => 
        s.id === sessionId ? session : s
      ))
      return session
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to end conversation')
      throw err
    }
  }, [])

  return {
    sessions,
    loading,
    error,
    startConversation,
    addMessage,
    endConversation,
  }
}

// Hook for real-time updates (placeholder for WebSocket integration)
export const useBrandAgentUpdates = (brandId: string) => {
  const [updates, setUpdates] = useState<any[]>([])

  useEffect(() => {
    // TODO: Implement WebSocket connection for real-time updates
    // const ws = new WebSocket(`ws://localhost:8000/ws/brand-agents/${brandId}`)
    
    // ws.onmessage = (event) => {
    //   const update = JSON.parse(event.data)
    //   setUpdates(prev => [...prev, update])
    // }

    // return () => {
    //   ws.close()
    // }
  }, [brandId])

  return {
    updates,
  }
}
