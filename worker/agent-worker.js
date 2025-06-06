/**
 * Cloudflare Worker for AI Agents with Durable Objects and Workflows
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    try {
      // Route requests
      if (url.pathname.startsWith('/api/agents/')) {
        return handleAgentRequest(request, env, ctx);
      } else if (url.pathname.startsWith('/api/workflows/')) {
        return handleWorkflowRequest(request, env, ctx);
      } else if (url.pathname.startsWith('/api/events/')) {
        return handleEventRequest(request, env, ctx);
      } else {
        return new Response('Cloudflare AI Agents Worker', {
          headers: { 'Content-Type': 'text/plain' }
        });
      }
    } catch (error) {
      console.error('Worker error:', error);
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }
};

/**
 * Handle agent-related requests
 */
async function handleAgentRequest(request, env, ctx) {
  const url = new URL(request.url);
  const agentId = url.pathname.split('/')[3];
  
  if (!agentId) {
    return new Response(JSON.stringify({ error: 'Agent ID required' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  // Get Durable Object for this agent
  const agentObjectId = env.AGENT_STATE.idFromName(agentId);
  const agentObject = env.AGENT_STATE.get(agentObjectId);
  
  // Forward request to Durable Object
  return agentObject.fetch(request);
}

/**
 * Handle workflow-related requests
 */
async function handleWorkflowRequest(request, env, ctx) {
  const url = new URL(request.url);
  const workflowId = url.pathname.split('/')[3];
  
  if (!workflowId) {
    return new Response(JSON.stringify({ error: 'Workflow ID required' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  // Get Durable Object for this workflow
  const workflowObjectId = env.WORKFLOW_ENGINE.idFromName(workflowId);
  const workflowObject = env.WORKFLOW_ENGINE.get(workflowObjectId);
  
  // Forward request to Durable Object
  return workflowObject.fetch(request);
}

/**
 * Handle event-related requests (waitForEvent API)
 */
async function handleEventRequest(request, env, ctx) {
  if (request.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  const body = await request.json();
  const { workflowId, eventType, payload } = body;
  
  if (!workflowId || !eventType) {
    return new Response(JSON.stringify({ error: 'Workflow ID and event type required' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  // Get workflow Durable Object
  const workflowObjectId = env.WORKFLOW_ENGINE.idFromName(workflowId);
  const workflowObject = env.WORKFLOW_ENGINE.get(workflowObjectId);
  
  // Send event to workflow
  const eventRequest = new Request('https://dummy.com/event', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ eventType, payload })
  });
  
  return workflowObject.fetch(eventRequest);
}

/**
 * Durable Object for Agent State Management
 */
export class AgentDurableObject {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.sessions = new Map();
  }
  
  async fetch(request) {
    const url = new URL(request.url);
    const method = request.method;
    
    try {
      if (method === 'GET' && url.pathname.endsWith('/state')) {
        return this.getAgentState();
      } else if (method === 'POST' && url.pathname.endsWith('/message')) {
        return this.addMessage(request);
      } else if (method === 'POST' && url.pathname.endsWith('/tool')) {
        return this.recordToolUsage(request);
      } else if (method === 'PUT' && url.pathname.endsWith('/context')) {
        return this.updateContext(request);
      } else {
        return new Response(JSON.stringify({ error: 'Not found' }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' }
        });
      }
    } catch (error) {
      console.error('Agent Durable Object error:', error);
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }
  
  async getAgentState() {
    const state = await this.state.storage.get('agentState') || {
      agentId: this.state.id.toString(),
      conversationHistory: [],
      contextVariables: {},
      toolUsageHistory: [],
      createdAt: new Date().toISOString(),
      lastActivity: new Date().toISOString()
    };
    
    return new Response(JSON.stringify(state), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async addMessage(request) {
    const { role, content, metadata } = await request.json();
    
    const state = await this.state.storage.get('agentState') || {
      conversationHistory: [],
      contextVariables: {},
      toolUsageHistory: []
    };
    
    const message = {
      role,
      content,
      timestamp: new Date().toISOString(),
      metadata: metadata || {}
    };
    
    state.conversationHistory.push(message);
    state.lastActivity = new Date().toISOString();
    
    await this.state.storage.put('agentState', state);
    
    return new Response(JSON.stringify({ success: true, message }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async recordToolUsage(request) {
    const { toolName, parameters, result } = await request.json();
    
    const state = await this.state.storage.get('agentState') || {
      conversationHistory: [],
      contextVariables: {},
      toolUsageHistory: []
    };
    
    const toolRecord = {
      toolName,
      parameters,
      result,
      timestamp: new Date().toISOString()
    };
    
    state.toolUsageHistory.push(toolRecord);
    state.lastActivity = new Date().toISOString();
    
    await this.state.storage.put('agentState', state);
    
    return new Response(JSON.stringify({ success: true, toolRecord }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async updateContext(request) {
    const { key, value } = await request.json();
    
    const state = await this.state.storage.get('agentState') || {
      conversationHistory: [],
      contextVariables: {},
      toolUsageHistory: []
    };
    
    state.contextVariables[key] = value;
    state.lastActivity = new Date().toISOString();
    
    await this.state.storage.put('agentState', state);
    
    return new Response(JSON.stringify({ success: true, key, value }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Durable Object for Workflow Engine with waitForEvent
 */
export class WorkflowDurableObject {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.pendingEvents = new Map();
    this.eventWaiters = new Map();
  }
  
  async fetch(request) {
    const url = new URL(request.url);
    const method = request.method;
    
    try {
      if (method === 'GET' && url.pathname.endsWith('/status')) {
        return this.getWorkflowStatus();
      } else if (method === 'POST' && url.pathname.endsWith('/start')) {
        return this.startWorkflow(request);
      } else if (method === 'POST' && url.pathname.endsWith('/event')) {
        return this.handleEvent(request);
      } else if (method === 'POST' && url.pathname.endsWith('/wait')) {
        return this.waitForEvent(request);
      } else {
        return new Response(JSON.stringify({ error: 'Not found' }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' }
        });
      }
    } catch (error) {
      console.error('Workflow Durable Object error:', error);
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }
  
  async getWorkflowStatus() {
    const workflow = await this.state.storage.get('workflow') || {
      workflowId: this.state.id.toString(),
      status: 'pending',
      steps: [],
      events: [],
      createdAt: new Date().toISOString()
    };
    
    return new Response(JSON.stringify(workflow), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async startWorkflow(request) {
    const { steps, context } = await request.json();
    
    const workflow = {
      workflowId: this.state.id.toString(),
      status: 'running',
      steps: steps || [],
      events: [],
      context: context || {},
      createdAt: new Date().toISOString(),
      startedAt: new Date().toISOString()
    };
    
    await this.state.storage.put('workflow', workflow);
    
    // Start workflow execution
    this.executeWorkflow(workflow);
    
    return new Response(JSON.stringify({ success: true, workflowId: workflow.workflowId }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async handleEvent(request) {
    const { eventType, payload } = await request.json();
    
    const event = {
      eventId: crypto.randomUUID(),
      eventType,
      payload,
      timestamp: new Date().toISOString()
    };
    
    // Store event
    const workflow = await this.state.storage.get('workflow') || { events: [] };
    workflow.events.push(event);
    await this.state.storage.put('workflow', workflow);
    
    // Check if any step is waiting for this event
    const waiters = this.eventWaiters.get(eventType) || [];
    for (const waiter of waiters) {
      waiter.resolve(event);
    }
    this.eventWaiters.delete(eventType);
    
    return new Response(JSON.stringify({ success: true, event }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async waitForEvent(request) {
    const { eventType, timeout = 300000 } = await request.json();
    
    // Check if event already exists
    const workflow = await this.state.storage.get('workflow') || { events: [] };
    const existingEvent = workflow.events.find(e => e.eventType === eventType);
    
    if (existingEvent) {
      return new Response(JSON.stringify({ success: true, event: existingEvent }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    // Wait for event with timeout
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error(`Timeout waiting for event: ${eventType}`));
      }, timeout);
      
      const waiter = {
        resolve: (event) => {
          clearTimeout(timeoutId);
          resolve(new Response(JSON.stringify({ success: true, event }), {
            headers: { 'Content-Type': 'application/json' }
          }));
        }
      };
      
      if (!this.eventWaiters.has(eventType)) {
        this.eventWaiters.set(eventType, []);
      }
      this.eventWaiters.get(eventType).push(waiter);
    });
  }
  
  async executeWorkflow(workflow) {
    // Simple workflow execution (in production, this would be more sophisticated)
    for (const step of workflow.steps) {
      try {
        if (step.type === 'waitForEvent') {
          // This step will wait for an event
          await this.waitForEvent({ json: () => ({ eventType: step.eventType }) });
        } else if (step.type === 'delay') {
          // Simple delay step
          await new Promise(resolve => setTimeout(resolve, step.duration || 1000));
        }
        // Add more step types as needed
      } catch (error) {
        console.error('Step execution error:', error);
        workflow.status = 'failed';
        workflow.error = error.message;
        await this.state.storage.put('workflow', workflow);
        return;
      }
    }
    
    workflow.status = 'completed';
    workflow.completedAt = new Date().toISOString();
    await this.state.storage.put('workflow', workflow);
  }
}
