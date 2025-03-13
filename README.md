# Agent Arcade Tools

<!-- [![CI](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml) -->

This is a proof of concept demonstrating an agent that combines [LangGraph](https://github.com/langchain-ai/langgraph) with [Arcade Tools](https://github.com/langchain-ai/langchain-arcade) and Azure Search to create an intelligent assistant capable of retrieving and processing instructional materials.

## Features

- Integration with LangGraph SDK for workflow management
- Arcade Tools integration for enhanced agent capabilities
- Azure OpenAI integration for language model capabilities
- Streamlit-based chat interface with real-time updates
- Support for tool calls and authorization flows
- Built-in debugging and tracing with LangSmith

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/yourusername/agent-arcade-tools.git
cd agent-arcade-tools
```

2. Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies using UV:

```bash
pip install uv
uv pip install -e .
```

4. Set up environment variables:

```bash
cp .env.example .env
```

Edit `.env` with your API keys and configuration:

- LangGraph API configuration
- Azure OpenAI or OpenAI API keys
- Arcade Tools API key
- User configuration
- (Optional) LangSmith configuration for debugging

5. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Architecture

The project uses a LangGraph-based architecture with the following components:

- `streamlit_app.py`: Main chat interface using Streamlit
- `src/agent_arcade_tools/`:
  - `graph.py`: LangGraph workflow definition
  - `configuration.py`: Configuration management
  - `state.py`: State management for the agent
  - `tools.py`: Tool implementations
  - `prompts.py`: System prompts and templates

## Development

The project uses modern Python development tools:

- `uv` for dependency management
- `pytest` for testing
- `ruff` for linting
- `mypy` for type checking

### Running Tests

```bash
python -m pytest tests/
```

### Type Checking

```bash
mypy src/
```

### Linting

```bash
ruff check src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and type checking
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for the workflow engine
- [Arcade Tools](https://github.com/langchain-ai/langchain-arcade) for enhanced agent capabilities
- [Streamlit](https://streamlit.io/) for the web interface
