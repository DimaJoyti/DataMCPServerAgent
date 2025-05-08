# Architecture

This document describes the architecture of the DataMCPServerAgent project.

## Overview

DataMCPServerAgent is a sophisticated agent system built on top of Bright Data MCP. It provides advanced agent architectures with memory persistence, tool selection, and learning capabilities.

## Agent Architectures

The project implements several agent architectures with increasing levels of sophistication:

### Basic Agent

The basic agent is a simple ReAct agent with Bright Data MCP tools. It uses the following components:

- **ChatAnthropic**: The language model used for generating responses
- **MCP Tools**: Tools for web automation and data collection
- **Custom Bright Data Tools**: Enhanced tools for specific tasks

### Advanced Agent

The advanced agent extends the basic agent with specialized sub-agents, tool selection, and memory. It uses the following components:

- **AgentMemory**: In-memory storage for conversation history, tool usage, and entities
- **ToolSelectionAgent**: Agent responsible for selecting the most appropriate tools for a task
- **SpecializedSubAgent**: Specialized agents for specific tasks (search, scraping, product research, etc.)
- **CoordinatorAgent**: Agent responsible for coordinating sub-agents

### Enhanced Agent

The enhanced agent adds memory persistence, enhanced tool selection, and learning capabilities. It uses the following components:

- **MemoryDatabase**: Database for persisting memory between sessions
- **EnhancedToolSelector**: Tool selector with performance tracking
- **FeedbackCollector**: Collector for user and self-evaluation feedback
- **LearningAgent**: Agent with learning capabilities
- **EnhancedCoordinatorAgent**: Coordinator agent with learning capabilities

### Advanced Enhanced Agent

The advanced enhanced agent adds context-aware memory, adaptive learning, and sophisticated tool selection. It uses the following components:

- **ContextManager**: Manager for maintaining and updating context
- **MemoryRetriever**: Retriever for relevant information from memory
- **AdaptiveLearningSystem**: System for adapting to user preferences
- **UserPreferenceModel**: Model for tracking and adapting to user preferences

### Multi-Agent Learning System

The multi-agent learning system extends the agent architecture with collaborative learning and knowledge sharing between agents. It uses the following components:

- **MultiAgentLearningSystem**: System for multi-agent learning and collaboration
- **CollaborativeLearningSystem**: System for collaborative learning between agents
- **KnowledgeTransferAgent**: Agent responsible for transferring knowledge between agents
- **CollaborativeKnowledgeBase**: Knowledge base for collaborative learning
- **AgentPerformanceTracker**: Tracker for agent performance metrics
- **MultiAgentPerformanceAnalyzer**: Analyzer for multi-agent performance

## Memory Architecture

The project implements several memory architectures:

### In-Memory Storage

The basic agent uses in-memory storage for conversation history, tool usage, and entities. This memory is lost when the agent is restarted.

### Persistent Storage

The enhanced agent uses persistent storage for memory. This allows the agent to remember past interactions and learn from them. The project supports the following storage backends:

- **SQLite**: Local database for memory persistence
- **File-Based**: File-based storage for memory persistence
- **Redis**: Distributed memory using Redis
- **MongoDB**: Distributed memory using MongoDB

### Context-Aware Memory

The advanced enhanced agent uses context-aware memory with semantic search. This allows the agent to retrieve relevant information based on the current request.

## Learning Architecture

The project implements several learning architectures:

### Feedback Collection

The enhanced agent collects feedback from users and performs self-evaluation. This feedback is used to improve future responses.

### Learning from Feedback

The learning agent analyzes feedback and develops strategies for improvement. It identifies patterns in feedback and suggests improvements.

### Adaptive Learning

The advanced enhanced agent adapts to user preferences. It tracks response style preferences, content preferences, tool preferences, and topic interests.

### Multi-Agent Learning

The multi-agent learning system enables agents to learn from each other's experiences. It includes:

- **Knowledge Extraction**: Extracting valuable knowledge from agent interactions
- **Knowledge Transfer**: Transferring knowledge between agents with different capabilities
- **Collaborative Problem-Solving**: Solving problems collaboratively using multiple agents
- **Performance Analysis**: Analyzing agent performance to identify opportunities for improvement
- **Synergy Identification**: Identifying synergistic combinations of agents for optimal performance

## Tool Selection Architecture

The project implements several tool selection architectures:

### Basic Tool Selection

The basic agent uses a simple tool selection algorithm based on the task description.

### Enhanced Tool Selection

The enhanced agent uses a sophisticated tool selection algorithm that considers task requirements, tool capabilities, historical performance, and user preferences.

### Performance Tracking

The enhanced agent tracks tool performance metrics such as success rate and execution time. This information is used to improve tool selection.

## Collaborative Knowledge Architecture

The project implements a collaborative knowledge architecture for multi-agent learning:

### Collaborative Knowledge Base

The collaborative knowledge base stores and manages knowledge shared between agents. It includes:

- **Knowledge Items**: Individual pieces of knowledge extracted from agent interactions
- **Knowledge Applicability**: Information about which agent types can benefit from each knowledge item
- **Knowledge Prerequisites**: Prerequisites for applying knowledge
- **Knowledge Transfers**: Records of knowledge transfers between agents
- **Agent Knowledge**: Knowledge assigned to each agent with proficiency levels

### Knowledge Transfer

The knowledge transfer system facilitates the sharing of knowledge between agents. It includes:

- **Knowledge Extraction**: Extracting valuable knowledge from agent interactions
- **Knowledge Adaptation**: Adapting knowledge for use by different agent types
- **Knowledge Application**: Applying knowledge to improve agent performance
- **Transfer Tracking**: Tracking the success of knowledge transfers

### Collaborative Problem-Solving

The collaborative problem-solving system enables agents to work together to solve complex problems. It includes:

- **Task Decomposition**: Breaking down complex tasks into subtasks
- **Agent Selection**: Selecting the most appropriate agents for each subtask
- **Result Synthesis**: Synthesizing results from multiple agents
- **Performance Tracking**: Tracking the performance of collaborative problem-solving

## Implementation Details

### Core Modules

- `src/core/main.py`: Basic agent entry point
- `src/core/advanced_main.py`: Advanced agent entry point
- `src/core/enhanced_main.py`: Enhanced agent entry point
- `src/core/advanced_enhanced_main.py`: Advanced enhanced agent entry point
- `src/core/multi_agent_main.py`: Multi-agent learning system entry point

### Agent Modules

- `src/agents/agent_architecture.py`: Base agent architecture with specialized sub-agents
- `src/agents/enhanced_agent_architecture.py`: Enhanced agent architecture with learning capabilities
- `src/agents/learning_capabilities.py`: Learning capabilities for agents
- `src/agents/adaptive_learning.py`: Adaptive learning system for user preferences
- `src/agents/multi_agent_learning.py`: Multi-agent learning system with collaborative capabilities

### Memory Modules

- `src/memory/memory_persistence.py`: Memory persistence using SQLite or files
- `src/memory/distributed_memory.py`: Distributed memory using Redis or MongoDB
- `src/memory/context_aware_memory.py`: Context-aware memory with semantic search
- `src/memory/collaborative_knowledge.py`: Collaborative knowledge base for multi-agent learning

### Tool Modules

- `src/tools/bright_data_tools.py`: Custom Bright Data MCP tools
- `src/tools/enhanced_tool_selection.py`: Enhanced tool selection with performance tracking

### Utility Modules

- `src/utils/agent_metrics.py`: Performance metrics for agents and multi-agent systems
- `src/utils/error_handlers.py`: Error handling utilities
- `src/utils/env_config.py`: Environment configuration utilities