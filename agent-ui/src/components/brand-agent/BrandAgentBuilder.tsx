'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Bot, 
  Brain, 
  MessageSquare, 
  Settings, 
  Zap, 
  Users, 
  ShoppingCart, 
  HeadphonesIcon,
  Sparkles,
  Target,
  BookOpen,
  Globe
} from 'lucide-react'

// Types
interface BrandAgent {
  id?: string
  name: string
  brandId: string
  agentType: string
  description: string
  personality: {
    traits: string[]
    tone: string
    communicationStyle: string
    responseLength: string
    formalityLevel: string
    emojiUsage: boolean
    customPhrases: string[]
  }
  configuration: {
    maxResponseLength: number
    responseTimeoutSeconds: number
    supportedChannels: string[]
    escalationTriggers: string[]
  }
  isActive: boolean
  isDeployed: boolean
}

const agentTypes = [
  {
    id: 'customer_support',
    name: 'Customer Support',
    description: 'Помогает клиентам с вопросами и проблемами',
    icon: HeadphonesIcon,
    color: 'bg-blue-500'
  },
  {
    id: 'sales_assistant',
    name: 'Sales Assistant',
    description: 'Помогает с продажами и консультациями',
    icon: ShoppingCart,
    color: 'bg-green-500'
  },
  {
    id: 'product_expert',
    name: 'Product Expert',
    description: 'Эксперт по продуктам и услугам',
    icon: Target,
    color: 'bg-purple-500'
  },
  {
    id: 'brand_ambassador',
    name: 'Brand Ambassador',
    description: 'Представляет бренд и его ценности',
    icon: Sparkles,
    color: 'bg-pink-500'
  },
  {
    id: 'content_creator',
    name: 'Content Creator',
    description: 'Создает контент и отвечает на вопросы',
    icon: BookOpen,
    color: 'bg-orange-500'
  },
  {
    id: 'lead_qualifier',
    name: 'Lead Qualifier',
    description: 'Квалифицирует лиды и потенциальных клиентов',
    icon: Users,
    color: 'bg-indigo-500'
  }
]

const personalityTraits = [
  'friendly', 'professional', 'enthusiastic', 'helpful', 
  'knowledgeable', 'empathetic', 'confident', 'creative', 
  'analytical', 'persuasive'
]

const communicationChannels = [
  { id: 'website_chat', name: 'Website Chat', icon: MessageSquare },
  { id: 'social_media', name: 'Social Media', icon: Globe },
  { id: 'email', name: 'Email', icon: Bot },
  { id: 'mobile_app', name: 'Mobile App', icon: Zap },
]

export const BrandAgentBuilder: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(0)
  const [agent, setAgent] = useState<BrandAgent>({
    name: '',
    brandId: '',
    agentType: '',
    description: '',
    personality: {
      traits: [],
      tone: 'professional',
      communicationStyle: 'helpful',
      responseLength: 'medium',
      formalityLevel: 'semi-formal',
      emojiUsage: false,
      customPhrases: []
    },
    configuration: {
      maxResponseLength: 500,
      responseTimeoutSeconds: 30,
      supportedChannels: [],
      escalationTriggers: []
    },
    isActive: true,
    isDeployed: false
  })

  const handleAgentTypeSelect = (typeId: string) => {
    setAgent(prev => ({ ...prev, agentType: typeId }))
  }

  const handlePersonalityTraitToggle = (trait: string) => {
    setAgent(prev => ({
      ...prev,
      personality: {
        ...prev.personality,
        traits: prev.personality.traits.includes(trait)
          ? prev.personality.traits.filter(t => t !== trait)
          : [...prev.personality.traits, trait].slice(0, 5) // Max 5 traits
      }
    }))
  }

  const handleChannelToggle = (channelId: string) => {
    setAgent(prev => ({
      ...prev,
      configuration: {
        ...prev.configuration,
        supportedChannels: prev.configuration.supportedChannels.includes(channelId)
          ? prev.configuration.supportedChannels.filter(c => c !== channelId)
          : [...prev.configuration.supportedChannels, channelId]
      }
    }))
  }

  const steps = [
    {
      title: 'Основная информация',
      description: 'Настройте базовые параметры агента'
    },
    {
      title: 'Тип агента',
      description: 'Выберите специализацию агента'
    },
    {
      title: 'Личность',
      description: 'Настройте характер и стиль общения'
    },
    {
      title: 'Каналы',
      description: 'Выберите каналы для развертывания'
    },
    {
      title: 'Предварительный просмотр',
      description: 'Проверьте настройки перед созданием'
    }
  ]

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium mb-2">Название агента</label>
              <input
                type="text"
                value={agent.name}
                onChange={(e) => setAgent(prev => ({ ...prev, name: e.target.value }))}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Например: Помощник по продажам"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">ID бренда</label>
              <input
                type="text"
                value={agent.brandId}
                onChange={(e) => setAgent(prev => ({ ...prev, brandId: e.target.value }))}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="brand-123"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Описание</label>
              <textarea
                value={agent.description}
                onChange={(e) => setAgent(prev => ({ ...prev, description: e.target.value }))}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={3}
                placeholder="Опишите назначение и функции агента..."
              />
            </div>
          </div>
        )

      case 1:
        return (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold mb-4">Выберите тип агента</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {agentTypes.map((type) => {
                const Icon = type.icon
                const isSelected = agent.agentType === type.id
                
                return (
                  <Card
                    key={type.id}
                    className={`cursor-pointer transition-all duration-200 hover:shadow-lg ${
                      isSelected 
                        ? 'ring-2 ring-blue-500 bg-blue-50 dark:bg-blue-950' 
                        : 'hover:shadow-md'
                    }`}
                    onClick={() => handleAgentTypeSelect(type.id)}
                  >
                    <CardContent className="p-6">
                      <div className="flex items-start space-x-4">
                        <div className={`w-12 h-12 rounded-lg ${type.color} flex items-center justify-center`}>
                          <Icon className="h-6 w-6 text-white" />
                        </div>
                        <div className="flex-1">
                          <h4 className="font-semibold text-lg">{type.name}</h4>
                          <p className="text-sm text-muted-foreground mt-1">{type.description}</p>
                        </div>
                        {isSelected && (
                          <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
                            <div className="w-2 h-2 bg-white rounded-full" />
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          </div>
        )

      case 2:
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-4">Черты характера</h3>
              <p className="text-sm text-muted-foreground mb-4">Выберите до 5 черт характера для вашего агента</p>
              <div className="flex flex-wrap gap-2">
                {personalityTraits.map((trait) => {
                  const isSelected = agent.personality.traits.includes(trait)
                  return (
                    <Badge
                      key={trait}
                      variant={isSelected ? "default" : "outline"}
                      className={`cursor-pointer transition-all duration-200 ${
                        isSelected 
                          ? 'bg-blue-500 hover:bg-blue-600' 
                          : 'hover:bg-blue-50 dark:hover:bg-blue-950'
                      }`}
                      onClick={() => handlePersonalityTraitToggle(trait)}
                    >
                      {trait}
                    </Badge>
                  )
                })}
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Выбрано: {agent.personality.traits.length}/5
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium mb-2">Тон общения</label>
                <select
                  value={agent.personality.tone}
                  onChange={(e) => setAgent(prev => ({
                    ...prev,
                    personality: { ...prev.personality, tone: e.target.value }
                  }))}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="professional">Профессиональный</option>
                  <option value="friendly">Дружелюбный</option>
                  <option value="casual">Неформальный</option>
                  <option value="enthusiastic">Энтузиастичный</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Длина ответов</label>
                <select
                  value={agent.personality.responseLength}
                  onChange={(e) => setAgent(prev => ({
                    ...prev,
                    personality: { ...prev.personality, responseLength: e.target.value }
                  }))}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="short">Короткие</option>
                  <option value="medium">Средние</option>
                  <option value="long">Длинные</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Уровень формальности</label>
                <select
                  value={agent.personality.formalityLevel}
                  onChange={(e) => setAgent(prev => ({
                    ...prev,
                    personality: { ...prev.personality, formalityLevel: e.target.value }
                  }))}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="formal">Формальный</option>
                  <option value="semi-formal">Полуформальный</option>
                  <option value="informal">Неформальный</option>
                </select>
              </div>

              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  id="emoji-usage"
                  checked={agent.personality.emojiUsage}
                  onChange={(e) => setAgent(prev => ({
                    ...prev,
                    personality: { ...prev.personality, emojiUsage: e.target.checked }
                  }))}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="emoji-usage" className="text-sm font-medium">
                  Использовать эмодзи в ответах
                </label>
              </div>
            </div>
          </div>
        )

      case 3:
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-4">Каналы развертывания</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Выберите каналы, где будет работать ваш агент
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {communicationChannels.map((channel) => {
                  const Icon = channel.icon
                  const isSelected = agent.configuration.supportedChannels.includes(channel.id)
                  
                  return (
                    <Card
                      key={channel.id}
                      className={`cursor-pointer transition-all duration-200 ${
                        isSelected 
                          ? 'ring-2 ring-blue-500 bg-blue-50 dark:bg-blue-950' 
                          : 'hover:shadow-md'
                      }`}
                      onClick={() => handleChannelToggle(channel.id)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-center space-x-3">
                          <Icon className="h-6 w-6 text-blue-500" />
                          <span className="font-medium">{channel.name}</span>
                          {isSelected && (
                            <div className="ml-auto w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center">
                              <div className="w-2 h-2 bg-white rounded-full" />
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium mb-2">Максимальная длина ответа</label>
                <input
                  type="number"
                  value={agent.configuration.maxResponseLength}
                  onChange={(e) => setAgent(prev => ({
                    ...prev,
                    configuration: { 
                      ...prev.configuration, 
                      maxResponseLength: parseInt(e.target.value) || 500 
                    }
                  }))}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  min="100"
                  max="2000"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Таймаут ответа (сек)</label>
                <input
                  type="number"
                  value={agent.configuration.responseTimeoutSeconds}
                  onChange={(e) => setAgent(prev => ({
                    ...prev,
                    configuration: { 
                      ...prev.configuration, 
                      responseTimeoutSeconds: parseInt(e.target.value) || 30 
                    }
                  }))}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  min="5"
                  max="120"
                />
              </div>
            </div>
          </div>
        )

      case 4:
        return (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold mb-4">Предварительный просмотр агента</h3>
            
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-3">
                  <Bot className="h-6 w-6 text-blue-500" />
                  <span>{agent.name || 'Новый агент'}</span>
                </CardTitle>
                <CardDescription>{agent.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Тип агента</h4>
                  <Badge variant="outline">
                    {agentTypes.find(t => t.id === agent.agentType)?.name || 'Не выбран'}
                  </Badge>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Черты характера</h4>
                  <div className="flex flex-wrap gap-1">
                    {agent.personality.traits.map(trait => (
                      <Badge key={trait} variant="secondary" className="text-xs">
                        {trait}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Каналы развертывания</h4>
                  <div className="flex flex-wrap gap-1">
                    {agent.configuration.supportedChannels.map(channelId => {
                      const channel = communicationChannels.find(c => c.id === channelId)
                      return (
                        <Badge key={channelId} variant="outline" className="text-xs">
                          {channel?.name}
                        </Badge>
                      )
                    })}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Тон:</span>
                    <span className="ml-2 font-medium">{agent.personality.tone}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Длина ответов:</span>
                    <span className="ml-2 font-medium">{agent.personality.responseLength}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Формальность:</span>
                    <span className="ml-2 font-medium">{agent.personality.formalityLevel}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Эмодзи:</span>
                    <span className="ml-2 font-medium">
                      {agent.personality.emojiUsage ? 'Да' : 'Нет'}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )

      default:
        return null
    }
  }

  const canProceed = () => {
    switch (currentStep) {
      case 0:
        return agent.name && agent.brandId
      case 1:
        return agent.agentType
      case 2:
        return agent.personality.traits.length > 0
      case 3:
        return agent.configuration.supportedChannels.length > 0
      default:
        return true
    }
  }

  const handleCreateAgent = async () => {
    try {
      // Here you would call the API to create the agent
      console.log('Creating agent:', agent)
      // For now, just show success message
      alert('Агент успешно создан!')
    } catch (error) {
      console.error('Error creating agent:', error)
      alert('Ошибка при создании агента')
    }
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Создание Brand Agent</h1>
        <p className="text-muted-foreground">
          Создайте AI-агента для взаимодействия с вашими клиентами
        </p>
      </div>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {steps.map((step, index) => (
            <div key={index} className="flex items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium ${
                  index <= currentStep
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-600'
                }`}
              >
                {index + 1}
              </div>
              <div className="ml-3 hidden md:block">
                <p className="text-sm font-medium">{step.title}</p>
                <p className="text-xs text-muted-foreground">{step.description}</p>
              </div>
              {index < steps.length - 1 && (
                <div className="w-16 h-px bg-gray-300 mx-4 hidden md:block" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Step Content */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>{steps[currentStep].title}</CardTitle>
          <CardDescription>{steps[currentStep].description}</CardDescription>
        </CardHeader>
        <CardContent>
          {renderStepContent()}
        </CardContent>
      </Card>

      {/* Navigation */}
      <div className="flex justify-between">
        <Button
          variant="outline"
          onClick={() => setCurrentStep(prev => Math.max(0, prev - 1))}
          disabled={currentStep === 0}
        >
          Назад
        </Button>

        <div className="space-x-2">
          {currentStep < steps.length - 1 ? (
            <Button
              onClick={() => setCurrentStep(prev => prev + 1)}
              disabled={!canProceed()}
            >
              Далее
            </Button>
          ) : (
            <Button
              onClick={handleCreateAgent}
              disabled={!canProceed()}
              className="bg-green-500 hover:bg-green-600"
            >
              Создать агента
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
