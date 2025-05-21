import { useEffect, useState, useRef, useCallback } from "react";
import { useAgent } from "agents/react";
import { useAgentChat } from "agents/ai-react";
import type { Message } from "@ai-sdk/react";
import type { tools } from "./tools";

// Component imports
import { Button } from "./components/button/Button";
import { Card } from "./components/card/Card";
import { Avatar } from "./components/avatar/Avatar";
import { Toggle } from "./components/toggle/Toggle";
import { Textarea } from "./components/textarea/Textarea";
import { MemoizedMarkdown } from "./components/memoized-markdown";
import { ToolInvocationCard } from "./components/tool-invocation-card/ToolInvocationCard";
import { SEODashboard } from "./components/seo-dashboard/SEODashboard";
import { VisualizationPanel } from "./components/visualization-panel/VisualizationPanel";

// Icon imports
import {
  Bug,
  Moon,
  Robot,
  Sun,
  Trash,
  PaperPlaneTilt,
  Stop,
  MagnifyingGlass,
  ChartLine,
  Gauge,
  Globe,
  ArrowsClockwise,
} from "@phosphor-icons/react";

// List of tools that require human confirmation
const toolsRequiringConfirmation: (keyof typeof tools)[] = [
  "scheduleReport",
];

// Main App component
export default function App() {
  // State for dark mode
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== "undefined") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    return false;
  });

  // State for API key check
  const [hasApiKey, setHasApiKey] = useState<boolean | null>(null);
  const [apiKeyError, setApiKeyError] = useState<string | null>(null);

  // State for dashboard visibility
  const [showDashboard, setShowDashboard] = useState(false);
  const [dashboardDomain, setDashboardDomain] = useState<string | null>(null);

  // State for visualization panel
  const [showVisualization, setShowVisualization] = useState(false);
  const [visualizationData, setVisualizationData] = useState<any>(null);

  // Reference to the message container
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Check if API key is set
  useEffect(() => {
    async function checkApiKey() {
      try {
        const response = await fetch("/check-api-keys");
        const data = await response.json();
        setHasApiKey(data.success);
        if (!data.success) {
          setApiKeyError("ANTHROPIC_API_KEY is not set. Please set it in your environment variables.");
        }
      } catch (error) {
        setHasApiKey(false);
        setApiKeyError("Error checking API key. Please make sure the server is running.");
      }
    }
    checkApiKey();
  }, []);

  // Set up dark mode
  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  // Scroll to bottom of messages
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  // Set up agent chat
  const { messages, input, handleInputChange, handleSubmit, isLoading, stop } =
    useAgentChat({
      api: "/agents/seo-agent",
      onResponse: scrollToBottom,
      onToolCall: (call) => {
        // If the tool requires confirmation, show the confirmation dialog
        if (
          toolsRequiringConfirmation.includes(call.name as keyof typeof tools)
        ) {
          return new Promise((resolve) => {
            // Show confirmation dialog
            // For now, just auto-confirm
            resolve(true);
          });
        }
        return true;
      },
      onToolExecute: (call) => {
        // Handle special tool executions
        if (call.name === "visualizationGenerator") {
          setVisualizationData(call.parameters);
          setShowVisualization(true);
        } else if (call.name === "seoAnalyzer" || call.name === "bulkAnalysis") {
          // Extract domain from URL or use domain parameter
          const domain = call.parameters.domain || 
            (call.parameters.url ? new URL(call.parameters.url).hostname : null);
          
          if (domain) {
            setDashboardDomain(domain);
            setShowDashboard(true);
          }
        }
        return true;
      },
    });

  // Handle form submission
  const onSubmit = useCallback(
    (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      handleSubmit(e);
    },
    [handleSubmit]
  );

  // Clear chat history
  const clearChat = useCallback(() => {
    window.location.reload();
  }, []);

  // Toggle dark mode
  const toggleDarkMode = useCallback(() => {
    setDarkMode((prev) => !prev);
  }, []);

  // Toggle dashboard
  const toggleDashboard = useCallback(() => {
    setShowDashboard((prev) => !prev);
  }, []);

  // Toggle visualization panel
  const toggleVisualization = useCallback(() => {
    setShowVisualization((prev) => !prev);
  }, []);

  // API Key error component
  const ApiKeyError = () => {
    if (hasApiKey === null) return null;
    if (hasApiKey) return null;

    return (
      <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
        <div className="bg-white dark:bg-neutral-800 p-6 rounded-lg shadow-xl max-w-md w-full">
          <h2 className="text-xl font-bold mb-4">API Key Error</h2>
          <p className="mb-4">{apiKeyError}</p>
          <p className="mb-4 text-sm">
            To use the SEO Agent, you need to set the ANTHROPIC_API_KEY environment variable.
          </p>
          <div className="flex justify-end">
            <Button onClick={() => setHasApiKey(true)}>
              Continue Anyway
            </Button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="h-[100vh] w-full p-4 flex justify-center items-center bg-fixed overflow-hidden">
      <ApiKeyError />
      <div className="h-[calc(100vh-2rem)] w-full mx-auto max-w-6xl flex flex-col shadow-xl rounded-md overflow-hidden relative border border-neutral-300 dark:border-neutral-800">
        {/* Header */}
        <div className="px-4 py-3 border-b border-neutral-300 dark:border-neutral-800 flex items-center gap-3 sticky top-0 z-10">
          <div className="flex items-center justify-center h-8 w-8">
            <Robot size={24} weight="duotone" className="text-blue-500" />
          </div>
          <div className="flex-1">
            <h1 className="text-lg font-semibold">SEO Agent</h1>
            <p className="text-xs text-neutral-500 dark:text-neutral-400">
              Powered by Claude 3.5 Sonnet
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleDashboard}
              title="Toggle SEO Dashboard"
            >
              <Gauge size={20} />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleVisualization}
              title="Toggle Visualization Panel"
            >
              <ChartLine size={20} />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={clearChat}
              title="Clear chat"
            >
              <Trash size={20} />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleDarkMode}
              title="Toggle theme"
            >
              {darkMode ? <Sun size={20} /> : <Moon size={20} />}
            </Button>
          </div>
        </div>

        {/* Main content area with chat and dashboard */}
        <div className="flex-1 flex overflow-hidden">
          {/* Chat messages */}
          <div className={`flex-1 overflow-y-auto p-4 ${showDashboard ? 'w-1/2' : 'w-full'}`}>
            <div className="space-y-4 pb-20">
              {messages.map((message, i) => (
                <div key={i} className="flex flex-col gap-3">
                  <div className="flex items-start gap-3">
                    <Avatar
                      name={message.role === "user" ? "You" : "SEO Agent"}
                      role={message.role}
                    />
                    <Card className="flex-1">
                      <MemoizedMarkdown content={message.content} />
                    </Card>
                  </div>
                  {message.toolCalls?.map((toolCall, j) => (
                    <ToolInvocationCard
                      key={j}
                      toolCall={toolCall}
                      requiresConfirmation={toolsRequiringConfirmation.includes(
                        toolCall.name as keyof typeof tools
                      )}
                    />
                  ))}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Dashboard panel */}
          {showDashboard && (
            <div className="w-1/2 border-l border-neutral-300 dark:border-neutral-800 overflow-y-auto">
              <SEODashboard domain={dashboardDomain} />
            </div>
          )}
        </div>

        {/* Visualization panel */}
        {showVisualization && (
          <div className="absolute inset-0 bg-white dark:bg-neutral-900 z-20 overflow-auto">
            <div className="p-4 flex justify-between items-center border-b border-neutral-300 dark:border-neutral-800">
              <h2 className="text-lg font-semibold">SEO Visualization</h2>
              <Button variant="ghost" size="icon" onClick={toggleVisualization}>
                <ArrowsClockwise size={20} />
              </Button>
            </div>
            <div className="p-4">
              <VisualizationPanel data={visualizationData} />
            </div>
          </div>
        )}

        {/* Input area */}
        <div className="p-4 border-t border-neutral-300 dark:border-neutral-800 sticky bottom-0 bg-white dark:bg-neutral-900 z-10">
          <form onSubmit={onSubmit} className="flex gap-2">
            <Textarea
              value={input}
              onChange={handleInputChange}
              placeholder="Ask about SEO analysis, keyword research, content optimization, and more..."
              className="flex-1 min-h-12 max-h-36"
            />
            <div className="flex flex-col gap-2">
              {isLoading ? (
                <Button
                  type="button"
                  onClick={stop}
                  variant="destructive"
                  size="icon"
                  title="Stop generating"
                >
                  <Stop size={20} weight="bold" />
                </Button>
              ) : (
                <Button type="submit" size="icon" title="Send message">
                  <PaperPlaneTilt size={20} weight="bold" />
                </Button>
              )}
              <Button
                type="button"
                variant="outline"
                size="icon"
                title="Search the web"
                onClick={() => {
                  // Implement web search functionality
                }}
              >
                <MagnifyingGlass size={20} />
              </Button>
              <Button
                type="button"
                variant="outline"
                size="icon"
                title="Analyze website"
                onClick={() => {
                  // Implement quick website analysis
                }}
              >
                <Globe size={20} />
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
