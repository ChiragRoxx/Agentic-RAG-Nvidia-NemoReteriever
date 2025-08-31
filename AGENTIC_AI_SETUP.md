# NVIDIA Agent Intelligence Toolkit Setup Guide

This guide will help you set up the NVIDIA Agent Intelligence Toolkit to enable Agentic AI mode in your RAG system.

## Prerequisites

Before installing the NVIDIA Agent Intelligence Toolkit, ensure you have:

1. **Python 3.8 or higher**
2. **Git and Git LFS** installed
3. **NVIDIA API Key** (from [build.nvidia.com](https://build.nvidia.com/))
4. **Basic dependencies** already installed from requirements.txt

## Installation Options

### Option 1: Quick Install with Pip (Recommended)

```bash
pip install aiqtoolkit[langchain]
```

### Option 2: Install from Source (Advanced)

1. Clone the repository:
```bash
git clone https://github.com/NVIDIA/AgentIQ.git aiqtoolkit
cd aiqtoolkit
```

2. Initialize submodules:
```bash
git submodule update --init --recursive
```

3. Download datasets:
```bash
git lfs install
git lfs fetch
git lfs pull
```

4. Create and activate virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

5. Install with all dependencies:
```bash
pip install -e .[langchain]
```

6. Verify installation:
```bash
aiq --help
aiq --version
```

## Configuration

1. **Set your NVIDIA API Key** in your `.env` file:
```
NVIDIA_API_KEY=your_nvidia_api_key_here
```

2. **Install additional dependencies** for web search (optional):
```bash
pip install duckduckgo-search
```

## Features Enabled by Agentic AI Mode

When you enable Agentic AI mode in the RAG system, you get:

### 🧠 **Multi-Step Reasoning**
- The AI agent can break down complex questions into smaller parts
- Performs sequential reasoning steps to build comprehensive answers
- Shows its thinking process transparently

### 🔍 **Enhanced Tool Usage**
- **Knowledge Base Search**: Deep search through your documents
- **Web Search**: Access to real-time information from the internet
- **Information Analysis**: Synthesis and pattern recognition

### 🚀 **Autonomous Decision Making**
- Automatically decides which tools to use for each question
- Combines information from multiple sources
- Provides structured, well-reasoned responses

### 📊 **Performance Monitoring**
- Tracks tool usage and performance
- Shows processing steps and reasoning
- Provides insights into agent behavior

## Usage in the RAG System

1. **Start the Streamlit application**:
```bash
streamlit run streamlit_app.py
```

2. **Enable Agentic AI Mode**:
   - Look for the "🚀 Agentic AI Mode" toggle in the sidebar
   - Toggle it ON to enable autonomous AI agents
   - The system will show enhanced capabilities and tool usage

3. **Ask Complex Questions**:
   - Try questions that require multiple steps or external information
   - Watch how the agent uses different tools
   - Explore the reasoning process in the expandable sections

## Example Questions for Agentic Mode

Try these types of questions to see the agentic capabilities:

### Multi-Step Analysis
```
"Compare hydrogen production methods mentioned in the documents and provide a comprehensive analysis of their efficiency and costs"
```

### Information Synthesis
```
"What are the key challenges for hydrogen adoption globally, and how do they relate to the solutions proposed in our documents?"
```

### Current Information Requests
```
"What are the latest developments in green hydrogen technology this year?"
```

## Troubleshooting

### Common Issues

1. **Import Error**: `aiqtoolkit` not found
   - Solution: Ensure you've installed the toolkit with `pip install aiqtoolkit[langchain]`

2. **API Key Error**: NVIDIA API connection failed
   - Solution: Check your `.env` file has the correct `NVIDIA_API_KEY`

3. **Web Search Not Working**: DuckDuckGo search errors
   - Solution: Install with `pip install duckduckgo-search`

4. **Agent Initialization Failed**: Various agent setup errors
   - Solution: The system will fallback to standard RAG mode automatically

### Performance Tips

1. **Enable Web Search Selectively**: Web search can be slower, use for questions requiring current information
2. **Use Comprehensive Mode**: For document-heavy questions, enable both Agentic and Comprehensive modes
3. **Monitor Tool Usage**: Check which tools are being used in the sidebar information

## System Requirements

- **Memory**: 4GB+ RAM recommended for optimal performance
- **Internet**: Required for web search functionality
- **API Limits**: Be aware of NVIDIA API rate limits for production use

## Support and Resources

- **NVIDIA Agent Intelligence Toolkit Docs**: [docs.nvidia.com/aiqtoolkit](https://docs.nvidia.com/aiqtoolkit)
- **GitHub Repository**: [github.com/NVIDIA/AgentIQ](https://github.com/NVIDIA/AgentIQ)
- **NVIDIA Developer Forums**: [forums.developer.nvidia.com](https://forums.developer.nvidia.com)

## Next Steps

After setting up Agentic AI mode:

1. Experiment with different types of questions
2. Monitor performance and tool usage
3. Customize agent capabilities based on your needs
4. Explore advanced configurations in the source code

The Agentic AI mode transforms your RAG system from a simple document Q&A tool into an intelligent research assistant capable of complex reasoning and multi-source information synthesis.
