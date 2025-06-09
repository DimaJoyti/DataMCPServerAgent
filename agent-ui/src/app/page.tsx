'use client'
import Sidebar from '@/components/playground/Sidebar/Sidebar'
import { ChatArea } from '@/components/playground/ChatArea'
import { Phase3Dashboard } from '@/components/phase3/Phase3Dashboard'
import { BrandAgentManager } from '@/components/brand-agent/BrandAgentManager'
import { Suspense, useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Brain, MessageSquare, Zap, Sparkles } from 'lucide-react'

export default function Home() {
  const [activeMode, setActiveMode] = useState<'playground' | 'phase3' | 'brand-agents'>('playground')
  const [isHydrated, setIsHydrated] = useState(false)

  useEffect(() => {
    setIsHydrated(true)
  }, [])

  if (!isHydrated) {
    return (
      <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg animate-pulse">
            <Brain className="h-8 w-8 text-white" />
          </div>
          <div className="space-y-2">
            <h2 className="text-xl font-bold bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-300 bg-clip-text text-transparent">
              Loading DataMCP Agent System
            </h2>
            <p className="text-slate-600 dark:text-slate-400">Initializing advanced AI agents...</p>
          </div>
          <div className="flex space-x-1 justify-center">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
            <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <Suspense fallback={
      <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg animate-pulse">
            <Brain className="h-8 w-8 text-white" />
          </div>
          <div className="space-y-2">
            <h2 className="text-xl font-bold bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-300 bg-clip-text text-transparent">
              Loading DataMCP Agent System
            </h2>
            <p className="text-slate-600 dark:text-slate-400">Initializing advanced AI agents...</p>
          </div>
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
            <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
          </div>
        </div>
      </div>
    }>
      <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
        {/* Mode Selector */}
        <div className="border-b border-border/50 backdrop-blur-sm bg-white/80 dark:bg-slate-900/80 p-6 shadow-sm">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
                  <Brain className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-300 bg-clip-text text-transparent">
                    DataMCP Agent System
                  </h1>
                  <p className="text-sm text-muted-foreground">Advanced AI Agent Platform</p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <Button
                  variant={activeMode === 'playground' ? 'default' : 'outline'}
                  size="lg"
                  onClick={() => setActiveMode('playground')}
                  className={`transition-all duration-200 ${
                    activeMode === 'playground'
                      ? 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 shadow-lg shadow-blue-500/25'
                      : 'hover:bg-blue-50 dark:hover:bg-slate-800 border-2'
                  }`}
                >
                  <MessageSquare className="mr-2 h-5 w-5" />
                  Playground
                </Button>
                <Button
                  variant={activeMode === 'phase3' ? 'default' : 'outline'}
                  size="lg"
                  onClick={() => setActiveMode('phase3')}
                  className={`transition-all duration-200 ${
                    activeMode === 'phase3'
                      ? 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 shadow-lg shadow-indigo-500/25'
                      : 'hover:bg-indigo-50 dark:hover:bg-slate-800 border-2'
                  }`}
                >
                  <Brain className="mr-2 h-5 w-5" />
                  Dashboard
                  <Badge variant="secondary" className="ml-2 bg-gradient-to-r from-emerald-500 to-teal-500 text-white border-0">
                    <Zap className="mr-1 h-3 w-3" />
                    New
                  </Badge>
                </Button>
                <Button
                  variant={activeMode === 'brand-agents' ? 'default' : 'outline'}
                  size="lg"
                  onClick={() => setActiveMode('brand-agents')}
                  className={`transition-all duration-200 ${
                    activeMode === 'brand-agents'
                      ? 'bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 shadow-lg shadow-purple-500/25'
                      : 'hover:bg-purple-50 dark:hover:bg-slate-800 border-2'
                  }`}
                >
                  <Sparkles className="mr-2 h-5 w-5" />
                  Brand Agents
                  <Badge variant="secondary" className="ml-2 bg-gradient-to-r from-orange-500 to-red-500 text-white border-0">
                    <Zap className="mr-1 h-3 w-3" />
                    Beta
                  </Badge>
                </Button>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  {activeMode === 'playground'
                    ? 'Interactive Agent Chat'
                    : activeMode === 'phase3'
                    ? 'Integrated Semantic Agents'
                    : 'Brand Agent Platform'
                  }
                </p>
                <p className="text-xs text-muted-foreground">
                  {activeMode === 'playground'
                    ? 'Real-time AI conversations'
                    : activeMode === 'phase3'
                    ? 'Advanced agent management'
                    : 'AI-powered customer engagement'
                  }
                </p>
              </div>
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-lg shadow-green-500/50"></div>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="h-[calc(100vh-100px)]">
          {activeMode === 'playground' ? (
            <div className="flex h-full">
              <Sidebar />
              <ChatArea />
            </div>
          ) : activeMode === 'phase3' ? (
            <div className="h-full">
              <Phase3Dashboard />
            </div>
          ) : (
            <div className="h-full">
              <BrandAgentManager />
            </div>
          )}
        </div>
      </div>
    </Suspense>
  )
}
