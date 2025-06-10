'use client'

import React, { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  Upload, 
  FileText, 
  FolderOpen, 
  Settings, 
  Play, 
  ArrowLeft,
  CheckCircle,
  AlertCircle,
  Infinity,
  Zap
} from 'lucide-react'
import { useInfiniteLoopStore } from '@/store'
import { type InfiniteLoopConfig, type InfiniteLoopExecutionRequest } from '@/types/infinite-loop'
import { infiniteLoopAPI } from '@/services/infinite-loop/api'

export const ConfigurationPanel: React.FC = () => {
  const {
    defaultConfig,
    setUIState,
    addError,
    addSession
  } = useInfiniteLoopStore()

  const [config, setConfig] = useState<InfiniteLoopConfig>(defaultConfig)
  const [specFile, setSpecFile] = useState<File | null>(null)
  const [outputDir, setOutputDir] = useState<string>('')
  const [iterationCount, setIterationCount] = useState<number | 'infinite'>(10)
  const [isValidating, setIsValidating] = useState(false)
  const [isStarting, setIsStarting] = useState(false)
  const [validationResult, setValidationResult] = useState<any>(null)
  
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setSpecFile(file)
    setIsValidating(true)
    setValidationResult(null)

    try {
      const result = await infiniteLoopAPI.validateSpecification(file)
      setValidationResult(result)
    } catch (error) {
      addError({
        code: 'VALIDATION_ERROR',
        message: 'Failed to validate specification file',
        details: error,
        timestamp: new Date().toISOString()
      })
    } finally {
      setIsValidating(false)
    }
  }

  const handleStartExecution = async () => {
    if (!specFile || !outputDir) {
      addError({
        code: 'MISSING_PARAMETERS',
        message: 'Please provide specification file and output directory',
        timestamp: new Date().toISOString()
      })
      return
    }

    setIsStarting(true)

    try {
      const request: InfiniteLoopExecutionRequest = {
        specFile,
        outputDir,
        count: iterationCount,
        config
      }

      const response = await infiniteLoopAPI.startExecution(request)
      
      if (response.success) {
        // Create session object and add to store
        const session = {
          sessionId: response.sessionId,
          config,
          executionState: response.executionState,
          waves: [],
          agents: [],
          qualityMetrics: {
            overallScore: 0,
            uniquenessScore: 0,
            validationScore: 0,
            consistencyScore: 0,
            innovationScore: 0,
            trends: []
          },
          performanceMetrics: response.statistics || {
            executionTimeSeconds: 0,
            totalIterations: 0,
            completedIterations: 0,
            failedIterations: 0,
            successRate: 0,
            averageIterationTime: 0,
            qualityScore: 0,
            wavesCompleted: 0,
            contextUsage: 0,
            memoryUsage: 0,
            throughput: 0
          },
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        }

        addSession(session)
        setUIState({ selectedView: 'dashboard' })
      } else {
        addError({
          code: 'EXECUTION_START_FAILED',
          message: response.error || 'Failed to start execution',
          timestamp: new Date().toISOString()
        })
      }
    } catch (error) {
      addError({
        code: 'EXECUTION_ERROR',
        message: 'Failed to start infinite loop execution',
        details: error,
        timestamp: new Date().toISOString()
      })
    } finally {
      setIsStarting(false)
    }
  }

  const handleBackToDashboard = () => {
    setUIState({ selectedView: 'dashboard' })
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Button
            variant="outline"
            onClick={handleBackToDashboard}
            className="flex items-center space-x-2"
          >
            <ArrowLeft className="h-4 w-4" />
            <span>Back to Dashboard</span>
          </Button>
          <div>
            <h2 className="text-2xl font-bold">Configure Infinite Loop</h2>
            <p className="text-muted-foreground">Set up your infinite agentic loop execution</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column - File Upload & Basic Settings */}
        <div className="space-y-6">
          {/* Specification File Upload */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <FileText className="h-5 w-5" />
                <span>Specification File</span>
              </CardTitle>
              <CardDescription>
                Upload your specification file to define the content generation requirements
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt,.md,.json,.yaml,.yml"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                <p className="text-sm text-muted-foreground mb-2">
                  Click to upload or drag and drop
                </p>
                <Button
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isValidating}
                >
                  {isValidating ? 'Validating...' : 'Choose File'}
                </Button>
              </div>

              {specFile && (
                <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                  <div className="flex items-center space-x-2">
                    <FileText className="h-4 w-4" />
                    <span className="text-sm font-medium">{specFile.name}</span>
                  </div>
                  <Badge variant="outline">
                    {(specFile.size / 1024).toFixed(1)} KB
                  </Badge>
                </div>
              )}

              {isValidating && (
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                    <span className="text-sm">Validating specification...</span>
                  </div>
                  <Progress value={undefined} className="w-full" />
                </div>
              )}

              {validationResult && (
                <div className={`p-3 rounded-lg ${validationResult.valid ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
                  <div className="flex items-center space-x-2 mb-2">
                    {validationResult.valid ? (
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-red-600" />
                    )}
                    <span className={`text-sm font-medium ${validationResult.valid ? 'text-green-800' : 'text-red-800'}`}>
                      {validationResult.valid ? 'Specification Valid' : 'Validation Failed'}
                    </span>
                  </div>
                  {validationResult.errors?.length > 0 && (
                    <ul className="text-sm text-red-700 space-y-1">
                      {validationResult.errors.map((error: string, index: number) => (
                        <li key={index}>• {error}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Output Directory */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <FolderOpen className="h-5 w-5" />
                <span>Output Directory</span>
              </CardTitle>
              <CardDescription>
                Specify where generated iterations will be saved
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <Label htmlFor="output-dir">Directory Path</Label>
                <Input
                  id="output-dir"
                  value={outputDir}
                  onChange={(e) => setOutputDir(e.target.value)}
                  placeholder="/path/to/output/directory"
                />
              </div>
            </CardContent>
          </Card>

          {/* Iteration Count */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Infinity className="h-5 w-5" />
                <span>Iteration Count</span>
              </CardTitle>
              <CardDescription>
                Set the number of iterations or enable infinite mode
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-4">
                <div className="flex-1">
                  <Label htmlFor="iteration-count">Number of Iterations</Label>
                  <Input
                    id="iteration-count"
                    type="number"
                    value={iterationCount === 'infinite' ? '' : iterationCount}
                    onChange={(e) => setIterationCount(parseInt(e.target.value) || 1)}
                    disabled={iterationCount === 'infinite'}
                    min={1}
                    max={1000}
                  />
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={iterationCount === 'infinite'}
                    onCheckedChange={(checked: boolean) => setIterationCount(checked ? 'infinite' : 10)}
                  />
                  <Label>Infinite Mode</Label>
                </div>
              </div>
              {iterationCount === 'infinite' && (
                <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Infinity className="h-4 w-4 text-blue-600" />
                    <span className="text-sm font-medium text-blue-800">Infinite Mode Enabled</span>
                  </div>
                  <p className="text-sm text-blue-700 mt-1">
                    The loop will continue until manually stopped or context limits are reached.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Advanced Configuration */}
        <div className="space-y-6">
          {/* Advanced Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Settings className="h-5 w-5" />
                <span>Advanced Configuration</span>
              </CardTitle>
              <CardDescription>
                Fine-tune the execution parameters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Parallel Agents */}
              <div className="space-y-2">
                <Label htmlFor="max-agents">Max Parallel Agents</Label>
                <Input
                  id="max-agents"
                  type="number"
                  value={config.maxParallelAgents}
                  onChange={(e) => setConfig(prev => ({ ...prev, maxParallelAgents: parseInt(e.target.value) || 5 }))}
                  min={1}
                  max={20}
                />
              </div>

              {/* Wave Size */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="wave-min">Min Wave Size</Label>
                  <Input
                    id="wave-min"
                    type="number"
                    value={config.waveSizeMin}
                    onChange={(e) => setConfig(prev => ({ ...prev, waveSizeMin: parseInt(e.target.value) || 3 }))}
                    min={1}
                    max={10}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="wave-max">Max Wave Size</Label>
                  <Input
                    id="wave-max"
                    type="number"
                    value={config.waveSizeMax}
                    onChange={(e) => setConfig(prev => ({ ...prev, waveSizeMax: parseInt(e.target.value) || 5 }))}
                    min={1}
                    max={20}
                  />
                </div>
              </div>

              {/* Quality Thresholds */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="quality-threshold">Quality Threshold</Label>
                  <Input
                    id="quality-threshold"
                    type="number"
                    step="0.1"
                    value={config.qualityThreshold}
                    onChange={(e) => setConfig(prev => ({ ...prev, qualityThreshold: parseFloat(e.target.value) || 0.7 }))}
                    min={0}
                    max={1}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="uniqueness-threshold">Uniqueness Threshold</Label>
                  <Input
                    id="uniqueness-threshold"
                    type="number"
                    step="0.1"
                    value={config.uniquenessThreshold}
                    onChange={(e) => setConfig(prev => ({ ...prev, uniquenessThreshold: parseFloat(e.target.value) || 0.8 }))}
                    min={0}
                    max={1}
                  />
                </div>
              </div>

              {/* Switches */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label htmlFor="validation">Enable Validation</Label>
                  <Switch
                    id="validation"
                    checked={config.validationEnabled}
                    onCheckedChange={(checked: boolean) => setConfig(prev => ({ ...prev, validationEnabled: checked }))}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <Label htmlFor="error-recovery">Error Recovery</Label>
                  <Switch
                    id="error-recovery"
                    checked={config.errorRecoveryEnabled}
                    onCheckedChange={(checked: boolean) => setConfig(prev => ({ ...prev, errorRecoveryEnabled: checked }))}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <Label htmlFor="batch-processing">Batch Processing</Label>
                  <Switch
                    id="batch-processing"
                    checked={config.batchProcessing}
                    onCheckedChange={(checked: boolean) => setConfig(prev => ({ ...prev, batchProcessing: checked }))}
                  />
                </div>
              </div>

              {/* Log Level */}
              <div className="space-y-2">
                <Label htmlFor="log-level">Log Level</Label>
                <Select
                  value={config.logLevel}
                  onValueChange={(value: any) => setConfig(prev => ({ ...prev, logLevel: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="DEBUG">Debug</SelectItem>
                    <SelectItem value="INFO">Info</SelectItem>
                    <SelectItem value="WARNING">Warning</SelectItem>
                    <SelectItem value="ERROR">Error</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Start Execution */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Zap className="h-5 w-5" />
                <span>Start Execution</span>
              </CardTitle>
              <CardDescription>
                Begin the infinite agentic loop with your configuration
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button
                onClick={handleStartExecution}
                disabled={!specFile || !outputDir || isStarting || (validationResult && !validationResult.valid)}
                className="w-full bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700"
                size="lg"
              >
                {isStarting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                    Starting Execution...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-5 w-5" />
                    Start Infinite Loop
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
