'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { 
  Bot, 
  Plus, 
  Settings, 
  Play, 
  Pause, 
  BarChart3, 
  MessageSquare, 
  Users, 
  TrendingUp,
  Eye,
  Edit,
  Trash2,
  Globe,
  Zap
} from 'lucide-react'

// Types
interface BrandAgent {
  id: string
  name: string
  brandId: string
  agentType: string
  description: string
  isActive: boolean
  isDeployed: boolean
  deploymentChannels: string[]
  successRate: number
  totalConversations: number
  averageSatisfaction: number
  createdAt: string
  updatedAt: string
}

interface BrandSummary {
  brandId: string
  totalAgents: number
  activeAgents: number
  deployedAgents: number
  totalConversations: number
  averageSatisfaction: number
  agents: BrandAgent[]
}

const agentTypeLabels: Record<string, string> = {
  customer_support: 'Поддержка клиентов',
  sales_assistant: 'Помощник по продажам',
  product_expert: 'Эксперт по продуктам',
  brand_ambassador: 'Амбассадор бренда',
  content_creator: 'Создатель контента',
  lead_qualifier: 'Квалификатор лидов'
}

const channelLabels: Record<string, string> = {
  website_chat: 'Чат на сайте',
  social_media: 'Социальные сети',
  email: 'Email',
  mobile_app: 'Мобильное приложение'
}

interface BrandAgentDashboardProps {
  brandId?: string
  onCreateAgent?: () => void
  onEditAgent?: (agentId: string) => void
}

export const BrandAgentDashboard: React.FC<BrandAgentDashboardProps> = ({
  brandId = 'demo-brand',
  onCreateAgent,
  onEditAgent
}) => {
  const [summary, setSummary] = useState<BrandSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)

  // Mock data for demonstration
  useEffect(() => {
    const mockSummary: BrandSummary = {
      brandId: brandId,
      totalAgents: 4,
      activeAgents: 3,
      deployedAgents: 2,
      totalConversations: 1247,
      averageSatisfaction: 4.3,
      agents: [
        {
          id: 'agent-1',
          name: 'Помощник по продажам',
          brandId: brandId,
          agentType: 'sales_assistant',
          description: 'Помогает клиентам с выбором продуктов и оформлением заказов',
          isActive: true,
          isDeployed: true,
          deploymentChannels: ['website_chat', 'mobile_app'],
          successRate: 87.5,
          totalConversations: 523,
          averageSatisfaction: 4.5,
          createdAt: '2024-01-15T10:00:00Z',
          updatedAt: '2024-01-20T15:30:00Z'
        },
        {
          id: 'agent-2',
          name: 'Поддержка клиентов',
          brandId: brandId,
          agentType: 'customer_support',
          description: 'Отвечает на вопросы клиентов и решает проблемы',
          isActive: true,
          isDeployed: true,
          deploymentChannels: ['website_chat', 'email'],
          successRate: 92.1,
          totalConversations: 724,
          averageSatisfaction: 4.2,
          createdAt: '2024-01-10T09:00:00Z',
          updatedAt: '2024-01-19T12:15:00Z'
        },
        {
          id: 'agent-3',
          name: 'Эксперт по продуктам',
          brandId: brandId,
          agentType: 'product_expert',
          description: 'Предоставляет детальную информацию о продуктах',
          isActive: true,
          isDeployed: false,
          deploymentChannels: [],
          successRate: 0,
          totalConversations: 0,
          averageSatisfaction: 0,
          createdAt: '2024-01-18T14:00:00Z',
          updatedAt: '2024-01-18T14:00:00Z'
        },
        {
          id: 'agent-4',
          name: 'Амбассадор бренда',
          brandId: brandId,
          agentType: 'brand_ambassador',
          description: 'Представляет бренд и его ценности',
          isActive: false,
          isDeployed: false,
          deploymentChannels: [],
          successRate: 0,
          totalConversations: 0,
          averageSatisfaction: 0,
          createdAt: '2024-01-12T11:00:00Z',
          updatedAt: '2024-01-16T16:45:00Z'
        }
      ]
    }

    setTimeout(() => {
      setSummary(mockSummary)
      setLoading(false)
    }, 1000)
  }, [brandId])

  const handleToggleAgent = async (agentId: string, isActive: boolean) => {
    // Mock API call
    console.log(`${isActive ? 'Activating' : 'Deactivating'} agent:`, agentId)
    
    if (summary) {
      const updatedAgents = summary.agents.map(agent =>
        agent.id === agentId ? { ...agent, isActive: !agent.isActive } : agent
      )
      setSummary({
        ...summary,
        agents: updatedAgents,
        activeAgents: updatedAgents.filter(a => a.isActive).length
      })
    }
  }

  const handleDeployAgent = async (agentId: string, channel: string) => {
    // Mock API call
    console.log(`Deploying agent ${agentId} to channel:`, channel)
  }

  const handleDeleteAgent = async (agentId: string) => {
    if (confirm('Вы уверены, что хотите удалить этого агента?')) {
      console.log('Deleting agent:', agentId)
      
      if (summary) {
        const updatedAgents = summary.agents.filter(agent => agent.id !== agentId)
        setSummary({
          ...summary,
          agents: updatedAgents,
          totalAgents: updatedAgents.length,
          activeAgents: updatedAgents.filter(a => a.isActive).length,
          deployedAgents: updatedAgents.filter(a => a.isDeployed).length
        })
      }
    }
  }

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded w-1/3"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-24 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (!summary) {
    return (
      <div className="p-6 text-center">
        <p className="text-muted-foreground">Не удалось загрузить данные</p>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Brand Agents Dashboard</h1>
          <p className="text-muted-foreground">
            Управляйте AI-агентами вашего бренда
          </p>
        </div>
        <Button onClick={onCreateAgent} className="bg-blue-500 hover:bg-blue-600">
          <Plus className="mr-2 h-4 w-4" />
          Создать агента
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Всего агентов</p>
                <p className="text-2xl font-bold">{summary.totalAgents}</p>
              </div>
              <Bot className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Активных</p>
                <p className="text-2xl font-bold text-green-600">{summary.activeAgents}</p>
              </div>
              <Zap className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Развернутых</p>
                <p className="text-2xl font-bold text-purple-600">{summary.deployedAgents}</p>
              </div>
              <Globe className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Разговоров</p>
                <p className="text-2xl font-bold">{summary.totalConversations.toLocaleString()}</p>
              </div>
              <MessageSquare className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Agents List */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Ваши агенты</h2>
        
        {summary.agents.length === 0 ? (
          <Card>
            <CardContent className="p-12 text-center">
              <Bot className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">Нет агентов</h3>
              <p className="text-muted-foreground mb-4">
                Создайте своего первого AI-агента для взаимодействия с клиентами
              </p>
              <Button onClick={onCreateAgent}>
                <Plus className="mr-2 h-4 w-4" />
                Создать первого агента
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {summary.agents.map((agent) => (
              <Card key={agent.id} className="hover:shadow-lg transition-shadow duration-200">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <CardTitle className="flex items-center space-x-2">
                        <Bot className="h-5 w-5 text-blue-500" />
                        <span>{agent.name}</span>
                      </CardTitle>
                      <CardDescription className="mt-1">
                        {agent.description}
                      </CardDescription>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant={agent.isActive ? "default" : "secondary"}>
                        {agent.isActive ? "Активен" : "Неактивен"}
                      </Badge>
                      {agent.isDeployed && (
                        <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                          Развернут
                        </Badge>
                      )}
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Тип:</span>
                    <span className="font-medium">
                      {agentTypeLabels[agent.agentType] || agent.agentType}
                    </span>
                  </div>

                  {agent.deploymentChannels.length > 0 && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-2">Каналы:</p>
                      <div className="flex flex-wrap gap-1">
                        {agent.deploymentChannels.map(channel => (
                          <Badge key={channel} variant="outline" className="text-xs">
                            {channelLabels[channel] || channel}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {agent.isDeployed && (
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div className="text-center">
                        <p className="text-muted-foreground">Успешность</p>
                        <p className="font-semibold text-green-600">
                          {agent.successRate.toFixed(1)}%
                        </p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground">Разговоров</p>
                        <p className="font-semibold">{agent.totalConversations}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground">Рейтинг</p>
                        <p className="font-semibold text-blue-600">
                          {agent.averageSatisfaction.toFixed(1)}/5
                        </p>
                      </div>
                    </div>
                  )}

                  <div className="flex items-center justify-between pt-4 border-t">
                    <div className="flex space-x-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleToggleAgent(agent.id, agent.isActive)}
                      >
                        {agent.isActive ? (
                          <>
                            <Pause className="mr-1 h-3 w-3" />
                            Остановить
                          </>
                        ) : (
                          <>
                            <Play className="mr-1 h-3 w-3" />
                            Запустить
                          </>
                        )}
                      </Button>

                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => onEditAgent?.(agent.id)}
                      >
                        <Edit className="mr-1 h-3 w-3" />
                        Изменить
                      </Button>
                    </div>

                    <div className="flex space-x-2">
                      <Button size="sm" variant="outline">
                        <BarChart3 className="mr-1 h-3 w-3" />
                        Аналитика
                      </Button>

                      <Button
                        size="sm"
                        variant="outline"
                        className="text-red-600 hover:text-red-700"
                        onClick={() => handleDeleteAgent(agent.id)}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
