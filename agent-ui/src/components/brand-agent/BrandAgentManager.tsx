'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Bot, 
  Plus, 
  BarChart3, 
  Settings, 
  BookOpen, 
  MessageSquare,
  ArrowLeft,
  Sparkles
} from 'lucide-react'

import { BrandAgentBuilder } from './BrandAgentBuilder'
import { BrandAgentDashboard } from './BrandAgentDashboard'
import { ChatTester } from './ChatTester'
import { AnalyticsDashboard } from '../analytics/AnalyticsDashboard'

type ViewMode = 'dashboard' | 'builder' | 'analytics' | 'knowledge' | 'conversations' | 'chat-tester'

type TimeRange = 'hour' | 'day' | 'week' | 'month'

interface BrandAgentManagerProps {
  brandId?: string
}

export const BrandAgentManager: React.FC<BrandAgentManagerProps> = ({
  brandId = 'demo-brand'
}) => {
  const [currentView, setCurrentView] = useState<ViewMode>('dashboard')
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null)
  const [timeRange, setTimeRange] = useState<TimeRange>('day')

  const handleCreateAgent = () => {
    setCurrentView('builder')
  }

  const handleEditAgent = (agentId: string) => {
    setSelectedAgentId(agentId)
    setCurrentView('builder')
  }

  const handleBackToDashboard = () => {
    setCurrentView('dashboard')
    setSelectedAgentId(null)
  }

  const renderCurrentView = () => {
    switch (currentView) {
      case 'builder':
        return (
          <div>
            <div className="mb-6">
              <Button
                variant="outline"
                onClick={handleBackToDashboard}
                className="mb-4"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Назад к панели управления
              </Button>
            </div>
            <BrandAgentBuilder />
          </div>
        )

      case 'analytics':
        return (
          <div>
            <div className="mb-6">
              <Button
                variant="outline"
                onClick={handleBackToDashboard}
                className="mb-4"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Назад к панели управления
              </Button>
            </div>

            <AnalyticsDashboard
              agentId={selectedAgentId || undefined}
              timeRange={timeRange}
              onTimeRangeChange={setTimeRange}
            />
          </div>
        )

      case 'knowledge':
        return (
          <div className="p-6">
            <div className="mb-6">
              <Button
                variant="outline"
                onClick={handleBackToDashboard}
                className="mb-4"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Назад к панели управления
              </Button>
            </div>
            
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-bold mb-2">База знаний</h1>
                  <p className="text-muted-foreground">
                    Управляйте знаниями ваших AI-агентов
                  </p>
                </div>
                <Button className="bg-green-500 hover:bg-green-600">
                  <Plus className="mr-2 h-4 w-4" />
                  Добавить знания
                </Button>
              </div>

              {/* Knowledge Base Content - Placeholder */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <BookOpen className="h-5 w-5 text-green-500" />
                      <span>Информация о продуктах</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground mb-4">
                      Подробные описания всех продуктов и услуг
                    </p>
                    <div className="flex justify-between text-sm">
                      <span>Статей: 45</span>
                      <span className="text-green-600">Активно</span>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <BookOpen className="h-5 w-5 text-blue-500" />
                      <span>FAQ</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground mb-4">
                      Часто задаваемые вопросы и ответы
                    </p>
                    <div className="flex justify-between text-sm">
                      <span>Вопросов: 127</span>
                      <span className="text-green-600">Активно</span>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <BookOpen className="h-5 w-5 text-purple-500" />
                      <span>Политики компании</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground mb-4">
                      Правила, политики и процедуры
                    </p>
                    <div className="flex justify-between text-sm">
                      <span>Документов: 23</span>
                      <span className="text-green-600">Активно</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        )

      case 'conversations':
        return (
          <div className="p-6">
            <div className="mb-6">
              <Button
                variant="outline"
                onClick={handleBackToDashboard}
                className="mb-4"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Назад к панели управления
              </Button>
            </div>

            <div className="space-y-6">
              <div>
                <h1 className="text-3xl font-bold mb-2">Разговоры</h1>
                <p className="text-muted-foreground">
                  Просматривайте и анализируйте разговоры ваших агентов
                </p>
              </div>

              {/* Conversations Content - Placeholder */}
              <Card>
                <CardHeader>
                  <CardTitle>Последние разговоры</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {[1, 2, 3, 4, 5].map((i) => (
                      <div key={i} className="flex items-center justify-between p-4 border rounded-lg">
                        <div className="flex items-center space-x-4">
                          <MessageSquare className="h-5 w-5 text-blue-500" />
                          <div>
                            <p className="font-medium">Разговор #{1000 + i}</p>
                            <p className="text-sm text-muted-foreground">
                              Помощник по продажам • 5 минут назад
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-4">
                          <span className="text-sm text-green-600">Завершен</span>
                          <span className="text-sm">⭐ 4.5</span>
                          <Button size="sm" variant="outline">
                            Просмотреть
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )

      case 'chat-tester':
        return (
          <div>
            <div className="mb-6">
              <Button
                variant="outline"
                onClick={handleBackToDashboard}
                className="mb-4"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Назад к панели управления
              </Button>
            </div>
            <ChatTester
              agents={[]} // This would come from your agent data
              onBack={handleBackToDashboard}
            />
          </div>
        )

      default:
        return (
          <BrandAgentDashboard
            brandId={brandId}
            onCreateAgent={handleCreateAgent}
            onEditAgent={handleEditAgent}
          />
        )
    }
  }

  if (currentView !== 'dashboard') {
    return renderCurrentView()
  }

  return (
    <div className="h-full">
      <Tabs defaultValue="dashboard" className="h-full">
        <div className="border-b border-border/50 backdrop-blur-sm bg-white/80 dark:bg-slate-900/80 p-6">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center shadow-lg">
                  <Sparkles className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-300 bg-clip-text text-transparent">
                    Brand Agent Platform
                  </h1>
                  <p className="text-sm text-muted-foreground">AI-powered customer engagement</p>
                </div>
              </div>
              
              <TabsList className="grid w-full grid-cols-6 lg:w-auto">
                <TabsTrigger value="dashboard" className="flex items-center space-x-2">
                  <Bot className="h-4 w-4" />
                  <span className="hidden sm:inline">Агенты</span>
                </TabsTrigger>
                <TabsTrigger
                  value="chat-tester"
                  className="flex items-center space-x-2"
                  onClick={() => setCurrentView('chat-tester')}
                >
                  <MessageSquare className="h-4 w-4" />
                  <span className="hidden sm:inline">Тест чат</span>
                </TabsTrigger>
                <TabsTrigger
                  value="analytics"
                  className="flex items-center space-x-2"
                  onClick={() => setCurrentView('analytics')}
                >
                  <BarChart3 className="h-4 w-4" />
                  <span className="hidden sm:inline">Аналитика</span>
                </TabsTrigger>
                <TabsTrigger
                  value="knowledge"
                  className="flex items-center space-x-2"
                  onClick={() => setCurrentView('knowledge')}
                >
                  <BookOpen className="h-4 w-4" />
                  <span className="hidden sm:inline">Знания</span>
                </TabsTrigger>
                <TabsTrigger
                  value="conversations"
                  className="flex items-center space-x-2"
                  onClick={() => setCurrentView('conversations')}
                >
                  <MessageSquare className="h-4 w-4" />
                  <span className="hidden sm:inline">Разговоры</span>
                </TabsTrigger>
                <TabsTrigger value="settings" className="flex items-center space-x-2">
                  <Settings className="h-4 w-4" />
                  <span className="hidden sm:inline">Настройки</span>
                </TabsTrigger>
              </TabsList>
            </div>
          </div>
        </div>

        <TabsContent value="dashboard" className="h-[calc(100vh-120px)] overflow-auto">
          <BrandAgentDashboard
            brandId={brandId}
            onCreateAgent={handleCreateAgent}
            onEditAgent={handleEditAgent}
          />
        </TabsContent>

        <TabsContent value="settings" className="h-[calc(100vh-120px)] overflow-auto p-6">
          <div className="max-w-4xl mx-auto space-y-6">
            <div>
              <h1 className="text-3xl font-bold mb-2">Настройки</h1>
              <p className="text-muted-foreground">
                Настройте параметры платформы Brand Agent
              </p>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Общие настройки</CardTitle>
                <CardDescription>
                  Основные параметры работы платформы
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Название бренда
                    </label>
                    <input
                      type="text"
                      defaultValue="Demo Brand"
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Часовой пояс
                    </label>
                    <select className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                      <option>UTC+3 (Москва)</option>
                      <option>UTC+0 (Лондон)</option>
                      <option>UTC-5 (Нью-Йорк)</option>
                    </select>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Уведомления</CardTitle>
                <CardDescription>
                  Настройте уведомления о работе агентов
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Email уведомления</p>
                    <p className="text-sm text-muted-foreground">
                      Получать уведомления на email
                    </p>
                  </div>
                  <input type="checkbox" defaultChecked className="w-4 h-4" />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Push уведомления</p>
                    <p className="text-sm text-muted-foreground">
                      Получать push уведомления в браузере
                    </p>
                  </div>
                  <input type="checkbox" className="w-4 h-4" />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
