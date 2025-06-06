'use client'

import React, { useState, useEffect } from 'react'
import { Phase3Navigation } from './Navigation/Phase3Navigation'
import { AgentDashboard } from './AgentDashboard/AgentDashboard'
import { PipelineVisualization } from './PipelineVisualization/PipelineVisualization'
import { TaskManagement } from './TaskManagement/TaskManagement'
import { PerformanceMonitoring } from './PerformanceMonitoring/PerformanceMonitoring'
import { AgentPlayground } from './AgentPlayground/AgentPlayground'

interface Phase3DashboardProps {
  className?: string
}

export const Phase3Dashboard: React.FC<Phase3DashboardProps> = ({ className }) => {
  const [activeView, setActiveView] = useState('dashboard')
  const [isHydrated, setIsHydrated] = useState(false)

  useEffect(() => {
    setIsHydrated(true)
  }, [])

  if (!isHydrated) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg animate-pulse">
            <div className="w-6 h-6 bg-white rounded-full"></div>
          </div>
          <p className="text-slate-600 dark:text-slate-400">Loading Dashboard...</p>
        </div>
      </div>
    )
  }

  const renderActiveView = () => {
    switch (activeView) {
      case 'dashboard':
        return <AgentDashboard />
      case 'pipelines':
        return <PipelineVisualization />
      case 'tasks':
        return <TaskManagement />
      case 'monitoring':
        return <PerformanceMonitoring />
      case 'playground':
        return <AgentPlayground />
      case 'settings':
        return (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-4">Settings</h2>
              <p className="text-muted-foreground">
                settings and configuration will be available here.
              </p>
            </div>
          </div>
        )
      default:
        return <AgentDashboard />
    }
  }

  return (
    <div className={`flex h-full ${className}`}>
      <Phase3Navigation
        activeView={activeView}
        onViewChange={setActiveView}
      />
      <main className="flex-1 overflow-auto bg-gradient-to-br from-slate-50/50 via-white to-blue-50/30 dark:from-slate-900/50 dark:via-slate-800 dark:to-slate-900">
        <div className="p-8 max-w-7xl mx-auto">
          <div className="animate-in fade-in-50 duration-500">
            {renderActiveView()}
          </div>
        </div>
      </main>
    </div>
  )
}
