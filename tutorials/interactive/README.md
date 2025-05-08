# Interactive Examples for DataMCPServerAgent

This directory contains interactive Jupyter notebooks and web-based examples for the DataMCPServerAgent.

## Jupyter Notebooks

The Jupyter notebooks provide an interactive environment for exploring and learning about the DataMCPServerAgent. Each notebook focuses on a specific aspect of the agent and includes code examples, explanations, and interactive elements.

### Getting Started

To run the Jupyter notebooks, you'll need to install the required dependencies:

```bash
pip install -r ../requirements.txt
```

Then, start the Jupyter notebook server:

```bash
jupyter notebook
```

This will open a browser window where you can navigate to and open the notebooks.

### Available Notebooks

1. **01_basic_agent.ipynb** - Introduction to the basic agent
2. **02_custom_tools.ipynb** - Creating and using custom tools
3. **03_memory_management.ipynb** - Working with the memory system
4. **04_tool_selection.ipynb** - Understanding the tool selection process
5. **05_multi_agent_system.ipynb** - Building a multi-agent system

## Web-Based Examples

The web-based examples use Voila to create interactive web applications based on Jupyter notebooks. These examples provide a more user-friendly interface for exploring the DataMCPServerAgent.

### Running Web-Based Examples

To run the web-based examples, you'll need to install Voila:

```bash
pip install voila
```

Then, run Voila with the notebook you want to convert to a web application:

```bash
voila web_app_basic_agent.ipynb
```

This will start a web server and open a browser window with the interactive application.

### Available Web Applications

1. **web_app_basic_agent.ipynb** - Interactive web application for the basic agent
2. **web_app_custom_tools.ipynb** - Interactive web application for creating and using custom tools
3. **web_app_memory_visualization.ipynb** - Interactive visualization of the memory system

## Creating Your Own Interactive Examples

You can create your own interactive examples by:

1. Creating a new Jupyter notebook in this directory
2. Adding code cells with examples of using the DataMCPServerAgent
3. Adding markdown cells with explanations
4. Adding interactive widgets using ipywidgets

For example, to create a simple interactive example with a text input and a button:

```python
import ipywidgets as widgets
from IPython.display import display
import asyncio
from src.core.main import chat_with_agent

# Create widgets
text_input = widgets.Text(
    value='',
    placeholder='Enter your message',
    description='Message:',
    disabled=False
)

output = widgets.Output()

button = widgets.Button(
    description='Send',
    disabled=False,
    button_style='', 
    tooltip='Send message to agent',
    icon='paper-plane'
)

# Define button click handler
async def on_button_clicked(b):
    with output:
        output.clear_output()
        print(f"User: {text_input.value}")
        response = await chat_with_agent(text_input.value)
        print(f"Agent: {response}")

# Use event loop to handle async function
def handle_button_click(b):
    asyncio.create_task(on_button_clicked(b))

button.on_click(handle_button_click)

# Display widgets
display(text_input, button, output)
```

This creates a simple interactive example where the user can enter a message, click the "Send" button, and see the agent's response.