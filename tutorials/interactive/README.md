# Interactive Tutorials

This directory contains comprehensive Jupyter notebooks that provide hands-on, interactive learning experiences for DataMCPServerAgent.

## 📚 Available Notebooks

### 🚀 Getting Started
- **`01_basic_agent.ipynb`** - Interactive introduction to the basic agent
  - Agent setup and configuration
  - Interactive chat interface
  - Tool exploration
  - Configuration customization

### 🧠 Advanced Agent Types
- **`02_advanced_agents.ipynb`** - Explore all agent architectures
  - Agent type comparison
  - Interactive configuration simulator
  - Performance metrics visualization
  - Best practices and recommendations

### 🏢 Enterprise Features
- **`03_data_pipelines.ipynb`** - Enterprise-grade capabilities
  - Data pipeline system exploration
  - Document processing simulator
  - Web interface testing
  - Monitoring and observability

### 🔧 Custom Development
- **`04_custom_tools.ipynb`** - Build your own tools
  - Interactive tool generator
  - Tool testing environment
  - Integration guide
  - Best practices and guidelines

## 🚀 Quick Start

### Prerequisites
```bash
# Install core dependencies
uv pip install -r requirements.txt

# Install tutorial dependencies (includes Jupyter)
uv pip install -r tutorials/requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running Notebooks

#### Option 1: Jupyter Notebook (Classic)
```bash
jupyter notebook tutorials/interactive/
```

#### Option 2: JupyterLab (Modern Interface)
```bash
jupyter lab tutorials/interactive/
```

#### Option 3: VS Code (If you have the Jupyter extension)
```bash
code tutorials/interactive/01_basic_agent.ipynb
```

## 📋 Learning Paths

### Path 1: Complete Beginner
1. **`01_basic_agent.ipynb`** - Start here for basic concepts
2. **`04_custom_tools.ipynb`** - Learn to create simple tools
3. Practice with the interactive examples

### Path 2: Agent Developer
1. Complete Path 1
2. **`02_advanced_agents.ipynb`** - Explore different agent types
3. **`03_data_pipelines.ipynb`** - Understand enterprise features
4. Build custom solutions

### Path 3: Enterprise Developer
1. Complete Path 1-2
2. Focus on **`03_data_pipelines.ipynb`** for production features
3. Implement monitoring and deployment strategies
4. Scale to production environments

## 🎯 Interactive Features

Each notebook includes:
- ✅ **Live Code Execution** - Run code directly in the browser
- ✅ **Interactive Widgets** - Buttons, sliders, and input fields
- ✅ **Real-time Output** - See results immediately
- ✅ **Configuration Tools** - Adjust settings on the fly
- ✅ **Progress Tracking** - Visual progress indicators
- ✅ **Error Handling** - Graceful error management
- ✅ **Best Practices** - Following development guidelines

## 🔧 Notebook Features

### Widget-Based Interfaces
- **Dropdown Selectors** - Choose options easily
- **Text Inputs** - Enter custom parameters
- **Sliders** - Adjust numerical values
- **Buttons** - Trigger actions and simulations
- **Output Areas** - View results and logs

### Interactive Simulations
- **Agent Configuration** - Test different settings
- **Performance Metrics** - Visualize agent performance
- **Tool Testing** - Validate custom tools
- **Pipeline Simulation** - Model data processing workflows

### Real-time Feedback
- **Instant Results** - See outputs immediately
- **Progress Indicators** - Track long-running operations
- **Error Messages** - Clear error reporting
- **Success Notifications** - Confirmation of completed tasks

## 🆘 Troubleshooting

### Common Issues

**Jupyter Not Starting:**
```bash
# Install Jupyter if missing
uv pip install jupyter

# Start with specific port
jupyter notebook --port=8888 tutorials/interactive/
```

**Widget Not Displaying:**
```bash
# Enable widgets extension
jupyter nbextension enable --py widgetsnbextension

# For JupyterLab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

**Kernel Issues:**
```bash
# Install kernel
python -m ipykernel install --user --name=datamcp

# Select the correct kernel in Jupyter
```

## 📈 What's Next?

After completing the interactive tutorials:

1. **Run Tutorial Scripts** - Try `tutorials/scripts/` for command-line learning
2. **Explore Examples** - Check `examples/` for real-world implementations
3. **Read Documentation** - Deep dive into `docs/` for detailed guides
4. **Build Custom Solutions** - Create your own agents and tools
5. **Join Community** - Share your work and get help

---

**Happy Interactive Learning! 🚀**

*DataMCPServerAgent - Empowering the next generation of AI agents*