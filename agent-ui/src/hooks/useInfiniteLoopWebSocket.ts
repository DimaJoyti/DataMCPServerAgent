import { useEffect, useCallback } from 'react'
import { useInfiniteLoopStore } from '@/store'
import { infiniteLoopWebSocket } from '@/services/infinite-loop/websocket'
import { type WebSocketMessage, type InfiniteLoopError } from '@/types/infinite-loop'

export const useInfiniteLoopWebSocket = (sessionId?: string) => {
  const {
    setIsConnected,
    setLastMessage,
    addError,
    updateSession,
    currentSession
  } = useInfiniteLoopStore()

  const handleMessage = useCallback((message: WebSocketMessage) => {
    setLastMessage(message)
    
    // Update session based on message type
    switch (message.type) {
      case 'execution_started':
        if (currentSession) {
          updateSession(currentSession.sessionId, {
            executionState: {
              ...currentSession.executionState,
              isRunning: true
            }
          })
        }
        break
        
      case 'wave_started':
        if (currentSession && message.data.waveNumber) {
          updateSession(currentSession.sessionId, {
            executionState: {
              ...currentSession.executionState,
              currentWave: message.data.waveNumber
            }
          })
        }
        break
        
      case 'wave_completed':
        if (currentSession && message.data.waveResult) {
          const updatedWaves = [...currentSession.waves, message.data.waveResult]
          updateSession(currentSession.sessionId, {
            waves: updatedWaves
          })
        }
        break
        
      case 'task_completed':
        if (currentSession && message.data.taskResult) {
          updateSession(currentSession.sessionId, {
            executionState: {
              ...currentSession.executionState,
              totalIterations: currentSession.executionState.totalIterations + 1,
              completedIterations: [
                ...currentSession.executionState.completedIterations,
                message.data.taskResult.iterationNumber.toString()
              ]
            }
          })
        }
        break
        
      case 'execution_completed':
        if (currentSession) {
          updateSession(currentSession.sessionId, {
            executionState: {
              ...currentSession.executionState,
              isRunning: false
            }
          })
        }
        break
        
      case 'metrics_update':
        if (currentSession && message.data.metrics) {
          updateSession(currentSession.sessionId, {
            performanceMetrics: message.data.metrics.performance || currentSession.performanceMetrics,
            qualityMetrics: message.data.metrics.quality || currentSession.qualityMetrics
          })
        }
        break
        
      case 'error':
        addError({
          code: message.data.code || 'WEBSOCKET_ERROR',
          message: message.data.message || 'WebSocket error occurred',
          details: message.data.details,
          timestamp: message.timestamp,
          sessionId: message.sessionId
        })
        break
    }
  }, [setLastMessage, addError, updateSession, currentSession])

  const handleError = useCallback((error: InfiniteLoopError) => {
    addError(error)
  }, [addError])

  const handleConnection = useCallback((connected: boolean) => {
    setIsConnected(connected)
  }, [setIsConnected])

  useEffect(() => {
    if (!sessionId) return

    // Set up event handlers
    const unsubscribeMessage = infiniteLoopWebSocket.onMessage(handleMessage)
    const unsubscribeError = infiniteLoopWebSocket.onError(handleError)
    const unsubscribeConnection = infiniteLoopWebSocket.onConnection(handleConnection)

    // Connect to WebSocket
    infiniteLoopWebSocket.connect(sessionId).catch((error) => {
      console.error('Failed to connect to WebSocket:', error)
      addError({
        code: 'WEBSOCKET_CONNECTION_FAILED',
        message: 'Failed to establish WebSocket connection',
        details: error,
        timestamp: new Date().toISOString(),
        sessionId
      })
    })

    // Cleanup on unmount
    return () => {
      unsubscribeMessage()
      unsubscribeError()
      unsubscribeConnection()
      infiniteLoopWebSocket.disconnect()
    }
  }, [sessionId, handleMessage, handleError, handleConnection, addError])

  return {
    isConnected: infiniteLoopWebSocket.isConnected(),
    send: infiniteLoopWebSocket.send.bind(infiniteLoopWebSocket),
    disconnect: infiniteLoopWebSocket.disconnect.bind(infiniteLoopWebSocket)
  }
}
