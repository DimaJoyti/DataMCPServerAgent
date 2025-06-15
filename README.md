# DataMCPServerAgent

A comprehensive AI agent system built with reinforcement learning, multi-agent coordination, and cloud integration capabilities. This project provides a modern, scalable platform for building intelligent agents that can learn, adapt, and collaborate to solve complex tasks.

## 🚀 Features

### Core Capabilities
- **Multi-Agent System**: Coordinate multiple specialized agents for complex tasks
- **Reinforcement Learning**: Advanced RL algorithms including DQN, PPO, and meta-learning
- **Cloud Integration**: Deploy and scale across AWS, Azure, and Google Cloud Platform
- **Real-time Communication**: WebSocket support for live agent interactions
- **Memory Systems**: Persistent and distributed memory with semantic search
- **Tool Integration**: Extensible tool system with performance tracking

### Advanced Features
- **Brand Agent Platform**: AI-powered conversational agents for marketing
- **Trading System**: Algorithmic trading with TradingView integration
- **Document Processing**: Advanced NLP pipeline with vector stores
- **Semantic Agents**: Context-aware agents with knowledge graphs
- **Infinite Loop System**: Continuous improvement and content generation

## 📋 Quick Start

### Prerequisites
- Python 3.9+
- Redis (optional, for distributed features)
- Node.js 18+ (for web UI)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/DataMCPServerAgent.git
cd DataMCPServerAgent

# Install Python dependencies
pip install -r requirements.txt

# Install UI dependencies
cd agent-ui
npm install
cd ..

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### Quick Start

```bash
# Start the API server
python app/main_consolidated.py api

# Start the web interface
cd agent-ui && npm run dev

# Access the application
# API: http://localhost:8003
# UI: http://localhost:3000
```

## 🏗️ Architecture

DataMCPServerAgent follows Clean Architecture principles with Domain-Driven Design:
```bash
# Start API server
python app/main_simple_consolidated.py api

# Or start CLI interface
python app/main_simple_consolidated.py cli
```

## 📖 Usage Examples

### API Server
```bash
# Start server with hot reload
python app/main_simple_consolidated.py api --reload

# Check system status
curl http://localhost:8003/health

# View API documentation
# Open http://localhost:8003/docs in your browser
```

### CLI Interface
```bash
# Interactive CLI
python app/main_simple_consolidated.py cli

# Available commands:
# - help: Show available commands
# - status: Show system status
# - agents: List available agents
# - tasks: Manage tasks
# - structure: Show system architecture
```

### Reinforcement Learning
```bash
# Basic RL mode
RL_MODE=basic python src/core/reinforcement_learning_main.py

# Advanced RL with modern algorithms
RL_MODE=modern_deep RL_ALGORITHM=ppo python src/core/reinforcement_learning_main.py

# Multi-agent learning
RL_MODE=multi_agent python src/core/reinforcement_learning_main.py
```

## 🏗️ Architecture

### System Structure
