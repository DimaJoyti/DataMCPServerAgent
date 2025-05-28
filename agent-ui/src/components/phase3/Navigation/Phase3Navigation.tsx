'use client'

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Brain,
  Activity,
  Settings,
  BarChart3,
  Zap,
  Image,
  Search,
  Timer,
  Eye,
  Menu,
  X
} from 'lucide-react'

interface NavigationItem {
  id: string
  label: string
  icon: React.ReactNode
  badge?: string
  active?: boolean
}

interface Phase3NavigationProps {
  activeView: string
  onViewChange: (view: string) => void
  className?: string
}

const navigationItems: NavigationItem[] = [
  {
    id: 'dashboard',
    label: 'Agent Dashboard',
    icon: <Brain className="h-5 w-5" />,
    badge: '3'
  },
  {
    id: 'pipelines',
    label: 'Pipeline Visualization',
    icon: <Activity className="h-5 w-5" />,
    badge: '2'
  },
  {
    id: 'tasks',
    label: 'Task Management',
    icon: <Timer className="h-5 w-5" />,
    badge: '5'
  },
  {
    id: 'monitoring',
    label: 'Performance Monitoring',
    icon: <BarChart3 className="h-5 w-5" />
  },
  {
    id: 'playground',
    label: 'Agent Playground',
    icon: <Zap className="h-5 w-5" />
  }
]

const agentTypes = [
  {
    id: 'multimodal',
    label: 'Multimodal Agents',
    icon: <Image className="h-4 w-4" />,
    count: 2,
    status: 'active'
  },
  {
    id: 'rag',
    label: 'RAG Agents',
    icon: <Search className="h-4 w-4" />,
    count: 1,
    status: 'busy'
  },
  {
    id: 'streaming',
    label: 'Streaming Agents',
    icon: <Zap className="h-4 w-4" />,
    count: 1,
    status: 'active'
  },
  {
    id: 'standard',
    label: 'Standard Agents',
    icon: <Brain className="h-4 w-4" />,
    count: 3,
    status: 'idle'
  }
]

const getStatusColor = (status: string) => {
  switch (status) {
    case 'active':
      return 'bg-green-500'
    case 'busy':
      return 'bg-yellow-500'
    case 'idle':
      return 'bg-gray-500'
    default:
      return 'bg-gray-500'
  }
}

export const Phase3Navigation: React.FC<Phase3NavigationProps> = ({
  activeView,
  onViewChange,
  className
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [isMobileOpen, setIsMobileOpen] = useState(false)

  return (
    <>
      {/* Mobile Menu Button */}
      <div className="lg:hidden fixed top-4 left-4 z-50">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsMobileOpen(!isMobileOpen)}
        >
          {isMobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </Button>
      </div>

      {/* Sidebar */}
      <div className={`
        ${className}
        ${isCollapsed ? 'w-16' : 'w-72'}
        ${isMobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        fixed lg:relative top-0 left-0 h-full
        bg-gradient-to-b from-white via-slate-50 to-slate-100
        dark:from-slate-900 dark:via-slate-800 dark:to-slate-900
        border-r border-border/50 backdrop-blur-sm
        transition-all duration-300 ease-in-out z-40
        flex flex-col shadow-xl
      `}>
        {/* Header */}
        <div className="p-6 border-b border-border/50">
          <div className="flex items-center justify-between">
            {!isCollapsed && (
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg">
                  <Brain className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="text-lg font-bold bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-300 bg-clip-text text-transparent">
                    Fun
                  </h2>
                  <p className="text-xs text-muted-foreground">Integrated Agents</p>
                </div>
              </div>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="hidden lg:flex hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg"
            >
              {isCollapsed ? <Menu className="h-4 w-4" /> : <X className="h-4 w-4" />}
            </Button>
          </div>
        </div>

        {/* Navigation Items */}
        <div className="flex-1 p-6 space-y-6">
          <div className="space-y-2">
            {!isCollapsed && (
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4 px-2">
                Main Navigation
              </h3>
            )}
            {navigationItems.map((item) => (
              <Button
                key={item.id}
                variant="ghost"
                className={`
                  w-full justify-start h-12 rounded-xl transition-all duration-200
                  ${isCollapsed ? 'px-3' : 'px-4'}
                  ${activeView === item.id
                    ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-500/25 hover:from-blue-600 hover:to-indigo-700'
                    : 'hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300'
                  }
                `}
                onClick={() => {
                  onViewChange(item.id)
                  setIsMobileOpen(false)
                }}
              >
                <div className="flex items-center space-x-3 w-full">
                  <div className={`${activeView === item.id ? 'text-white' : 'text-slate-500'}`}>
                    {item.icon}
                  </div>
                  {!isCollapsed && (
                    <>
                      <span className="flex-1 text-left font-medium">{item.label}</span>
                      {item.badge && (
                        <Badge
                          variant="secondary"
                          className={`ml-auto ${
                            activeView === item.id
                              ? 'bg-white/20 text-white border-0'
                              : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
                          }`}
                        >
                          {item.badge}
                        </Badge>
                      )}
                    </>
                  )}
                </div>
              </Button>
            ))}
          </div>

          {/* Agent Types */}
          <div className="space-y-2">
            {!isCollapsed && (
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4 px-2">
                Agent Types
              </h3>
            )}
            {agentTypes.map((type) => (
              <div
                key={type.id}
                className={`
                  flex items-center space-x-3 p-3 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800
                  cursor-pointer transition-all duration-200 group
                  ${isCollapsed ? 'justify-center' : ''}
                `}
                onClick={() => {
                  onViewChange('dashboard')
                  setIsMobileOpen(false)
                }}
              >
                <div className="text-slate-500 group-hover:text-slate-700 dark:group-hover:text-slate-300">
                  {type.icon}
                </div>
                {!isCollapsed && (
                  <>
                    <span className="flex-1 text-sm font-medium text-slate-700 dark:text-slate-300">
                      {type.label}
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-medium text-slate-500 bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded-full">
                        {type.count}
                      </span>
                      <div className={`w-2 h-2 rounded-full ${getStatusColor(type.status)} shadow-lg`} />
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-border/50">
          <Button
            variant="ghost"
            className={`w-full justify-start h-12 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 transition-all duration-200 ${isCollapsed ? 'px-3' : 'px-4'}`}
            onClick={() => {
              onViewChange('settings')
              setIsMobileOpen(false)
            }}
          >
            <div className="flex items-center space-x-3">
              <Settings className="h-5 w-5 text-slate-500" />
              {!isCollapsed && <span className="font-medium text-slate-700 dark:text-slate-300">Settings</span>}
            </div>
          </Button>

          {!isCollapsed && (
            <div className="mt-4 p-4 bg-gradient-to-br from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
              <div className="flex items-center space-x-2 mb-3">
                <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                  <Eye className="h-3 w-3 text-white" />
                </div>
                <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">System Status</span>
              </div>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between items-center">
                  <span className="text-slate-600 dark:text-slate-400">Agents Active:</span>
                  <span className="font-semibold text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/30 px-2 py-1 rounded-full">6/7</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-600 dark:text-slate-400">Tasks Running:</span>
                  <span className="font-semibold text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 px-2 py-1 rounded-full">3</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-600 dark:text-slate-400">System Load:</span>
                  <span className="font-semibold text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/30 px-2 py-1 rounded-full">45%</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Mobile Overlay */}
      {isMobileOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-30"
          onClick={() => setIsMobileOpen(false)}
        />
      )}
    </>
  )
}
