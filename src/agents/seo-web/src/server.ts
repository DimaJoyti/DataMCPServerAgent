import { routeAgentRequest, type Schedule } from "agents";
import { AIChatAgent } from "agents/ai-chat-agent";
import {
  createDataStreamResponse,
  generateId,
  streamText,
  type StreamTextOnFinishCallback,
  type ToolSet,
} from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { processToolCalls } from "./utils";
import { tools, executions } from "./tools";

// Initialize the Anthropic model
const model = anthropic("claude-3-5-sonnet-20240620");

// Define environment interface
interface Env {
  ANTHROPIC_API_KEY: string;
  SEMRUSH_API_KEY?: string;
  MOZ_ACCESS_ID?: string;
  MOZ_SECRET_KEY?: string;
  GOOGLE_SEARCH_CONSOLE_API_KEY?: string;
  AHREFS_API_KEY?: string;
  MAJESTIC_API_KEY?: string;
}

/**
 * SEO Agent implementation using Cloudflare's Agent platform
 */
export class SEOAgent extends AIChatAgent {
  constructor(env: Env) {
    super({
      name: "seo-agent",
      description: "SEO Agent for analyzing websites and providing optimization recommendations",
      systemPrompt: `You are an expert SEO agent specialized in search engine optimization.
Your goal is to help users improve their website's visibility in search engines and drive more organic traffic.

You can perform the following tasks:
1. Analyze websites for SEO factors
2. Research keywords and provide recommendations
3. Optimize content for better search engine rankings
4. Generate SEO-friendly metadata
5. Analyze backlink profiles
6. Analyze competitors and compare websites
7. Track keyword rankings over time
8. Perform bulk analysis of multiple pages or entire websites
9. Generate visualizations of SEO metrics
10. Schedule regular SEO reports

When responding to requests:
- Break down complex SEO tasks into manageable steps
- Provide clear explanations of SEO concepts
- Support your recommendations with data
- Prioritize recommendations based on impact and effort
- Focus on white-hat SEO techniques that follow search engine guidelines

Always consider the latest SEO best practices, including:
- Mobile-first indexing
- Page speed optimization
- User experience signals
- E-A-T (Expertise, Authoritativeness, Trustworthiness)
- Core Web Vitals
- Content quality and relevance
- Competitive analysis and differentiation`,
      tools,
      env,
    });
  }

  /**
   * Process a chat message from the user
   */
  async processMessage(message: string, env: Env): Promise<Response> {
    const chatId = generateId();
    const toolset: ToolSet = {};

    // Add all tools to the toolset
    for (const [name, tool] of Object.entries(tools)) {
      toolset[name] = tool;
    }

    // Create a stream for the response
    const stream = await streamText({
      model,
      messages: [
        { role: "system", content: this.systemPrompt },
        ...this.getHistory(),
        { role: "user", content: message },
      ],
      toolset,
      onToolCall: async (call) => {
        // Process tool calls
        return processToolCalls(call, executions, env);
      },
      onFinish: (async (finish) => {
        // Add the message to history
        this.addMessageToHistory({ role: "user", content: message });
        this.addMessageToHistory({ role: "assistant", content: finish.message.content });
      }) as StreamTextOnFinishCallback,
    });

    return createDataStreamResponse(stream);
  }

  /**
   * Process a scheduled task
   */
  async processScheduledTask(schedule: Schedule, env: Env): Promise<void> {
    // Handle scheduled tasks like regular SEO reports
    const { task, metadata } = schedule;
    
    console.log(`Processing scheduled task: ${task}`);
    
    // Execute the appropriate tool based on the task
    if (task === "generate_seo_report") {
      const domain = metadata?.domain as string;
      if (domain) {
        // Execute the bulk analysis tool
        const result = await executions.bulkAnalysis({
          domain,
          max_pages: 50,
          depth: "detailed",
          format: "markdown"
        }, env);
        
        // Store the report result
        console.log(`Generated SEO report for ${domain}`);
        
        // In a real implementation, we would store the report in a database
        // or send it via email to the user
      }
    }
  }
}

/**
 * Worker entry point that routes incoming requests to the appropriate handler
 */
export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext) {
    const url = new URL(request.url);

    if (url.pathname === "/check-api-keys") {
      // Check if required API keys are set
      const hasAnthropicKey = !!env.ANTHROPIC_API_KEY;
      const hasSEMrushKey = !!env.SEMRUSH_API_KEY;
      const hasMozKeys = !!env.MOZ_ACCESS_ID && !!env.MOZ_SECRET_KEY;
      
      return Response.json({
        success: hasAnthropicKey,
        seo_apis: {
          semrush: hasSEMrushKey,
          moz: hasMozKeys,
          google_search_console: !!env.GOOGLE_SEARCH_CONSOLE_API_KEY,
          ahrefs: !!env.AHREFS_API_KEY,
          majestic: !!env.MAJESTIC_API_KEY
        }
      });
    }
    
    if (!env.ANTHROPIC_API_KEY) {
      console.error(
        "ANTHROPIC_API_KEY is not set, don't forget to set it locally in .dev.vars, and use `wrangler secret bulk .dev.vars` to upload it to production"
      );
    }
    
    return (
      // Route the request to our agent or return 404 if not found
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  },
} satisfies ExportedHandler<Env>;
