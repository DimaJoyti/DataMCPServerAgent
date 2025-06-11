# Tutorial Scripts

This directory contains comprehensive tutorial scripts that demonstrate the full capabilities of DataMCPServerAgent.

## 📚 Available Scripts

### 🚀 Getting Started (Beginner)
- **`01_getting_started.py`** - Basic introduction to the agent system
  - Environment setup and configuration
  - Basic agent interaction
  - Command overview and usage
  - First steps with tools

### 🧠 Agent Types (Intermediate)
- **`02_agent_types.py`** - Comprehensive overview of all agent architectures
  - Basic Agent - Simple ReAct agent
  - Advanced Agent - Specialized sub-agents
  - Enhanced Agent - Memory and learning
  - Multi-Agent System - Collaborative agents
  - Reinforcement Learning Agent - Adaptive learning
  - Distributed Memory Agent - Scalable architecture
  - Knowledge Graph Agent - Entity modeling
  - Error Recovery Agent - Self-healing capabilities
  - Orchestration Agent - Advanced planning

### 🏢 Enterprise Features (Advanced)
- **`03_enterprise_features.py`** - Production-ready capabilities
  - Data Pipeline System - ETL/ELT workflows
  - Document Processing - Multi-format analysis
  - Web Interfaces - REST API and WebSocket
  - Monitoring & Observability - Performance tracking
  - Deployment strategies and best practices

### 🎯 Specialized Applications (Expert)
- **`04_specialized_apps.py`** - Domain-specific implementations
  - Research Assistant - Academic research tools
  - Trading Systems - Algorithmic trading and market analysis
  - Security & Penetration Testing - Automated security assessments
  - Marketing & SEO - Digital marketing automation
  - Custom application development

## 🚀 Quick Start

### Prerequisites
```bash
# Install core dependencies
uv pip install -r requirements.txt

# Install tutorial dependencies
uv pip install -r tutorials/requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running Scripts

#### 🎯 New to AI Agents?
Start with the basics:
```bash
python tutorials/scripts/01_getting_started.py
```

#### 🧠 Want to Explore Different Agents?
Learn about agent types:
```bash
python tutorials/scripts/02_agent_types.py
```

#### 🏢 Building Enterprise Solutions?
Explore advanced features:
```bash
python tutorials/scripts/03_enterprise_features.py
```

#### 🎯 Specific Use Cases?
Jump to specialized applications:
```bash
python tutorials/scripts/04_specialized_apps.py
```

## 📋 Learning Paths

### Path 1: Complete Beginner (2-3 hours)
1. `01_getting_started.py` - Basic setup and first agent
2. Run `examples/basic_agent_example.py` for hands-on practice
3. Try `examples/custom_tool_example.py` to understand tools

### Path 2: Agent Developer (4-6 hours)
1. Complete Path 1
2. `02_agent_types.py` - Understanding different agents
3. Run examples for each agent type
4. Experiment with configurations

### Path 3: Enterprise Developer (6-8 hours)
1. Complete Path 2
2. `03_enterprise_features.py` - Production features
3. Set up data pipelines and web interfaces
4. Configure monitoring and deployment

### Path 4: Domain Specialist (3-4 hours per domain)
1. Complete Path 1-2
2. `04_specialized_apps.py` - Choose your domain
3. Deep dive into domain-specific examples
4. Build custom solutions

## 🛠️ Script Features

Each tutorial script includes:
- ✅ **Step-by-step guidance** - Clear progression through concepts
- ✅ **Interactive examples** - Hands-on demonstrations
- ✅ **Error handling** - Graceful handling of common issues
- ✅ **Best practices** - Following development guidelines
- ✅ **Next steps** - Clear path to continue learning

## 🔧 Customization

### Environment Variables
Make sure these are set in your `.env` file:
```bash
ANTHROPIC_API_KEY=your_anthropic_key
BRIGHT_DATA_MCP_KEY=your_bright_data_key
# Optional for advanced features:
REDIS_URL=redis://localhost:6379
MONGODB_URL=mongodb://localhost:27017
```

### Configuration Options
Most scripts accept command-line arguments:
```bash
python tutorials/scripts/01_getting_started.py --verbose --model claude-3-opus
python tutorials/scripts/02_agent_types.py --demo-mode
python tutorials/scripts/03_enterprise_features.py --skip-setup
```

## 🆘 Troubleshooting

### Common Issues

**Environment Variables Not Set:**
```bash
# Check if variables are set
echo $ANTHROPIC_API_KEY
echo $BRIGHT_DATA_MCP_KEY

# Set them if missing
export ANTHROPIC_API_KEY="your_key_here"
export BRIGHT_DATA_MCP_KEY="your_key_here"
```

**Import Errors:**
```bash
# Make sure you're in the project root
cd /path/to/DataMCPServerAgent

# Install dependencies
uv pip install -r requirements.txt
uv pip install -r tutorials/requirements.txt
```

**Permission Errors:**
```bash
# Make scripts executable
chmod +x tutorials/scripts/*.py
```

## 📈 What's Next?

After completing the tutorial scripts:

1. **Interactive Notebooks** - Try `tutorials/interactive/` for hands-on learning
2. **Example Projects** - Explore `examples/` for real-world implementations
3. **Documentation** - Deep dive into `docs/` for detailed guides
4. **Custom Development** - Build your own agents and tools
5. **Community** - Share your creations and get help

## 🤝 Contributing

Help improve the tutorials:

1. **Report Issues** - Found a bug or unclear instruction?
2. **Suggest Improvements** - Ideas for better explanations?
3. **Add Examples** - Create new tutorial scenarios
4. **Share Use Cases** - Show how you're using the system

---

**Happy Learning! 🚀**

*DataMCPServerAgent - Empowering the next generation of AI agents*