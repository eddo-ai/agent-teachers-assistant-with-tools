# Agent Arcade Tools

<!-- [![CI](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml) -->

This is a proof of concept demonstrating an agent that combines [LangGraph](https://github.com/langchain-ai/langgraph) with [Arcade Tools](https://github.com/langchain-ai/langchain-arcade) and Azure Search to create an intelligent assistant capable of retrieving and processing instructional materials.

## What it does

The agent:

1. Takes a user **query** as input
2. Uses Azure Search to find relevant instructional materials
3. Processes the query using Arcade Tools for enhanced capabilities
4. Returns relevant information and materials to the user

## Features

- Integration with Azure Search for semantic search capabilities
- Arcade Tools integration for enhanced agent capabilities
- Asynchronous processing for better performance
- Configurable system prompts and model selection
- Built with LangGraph for flexible workflow management

## Getting Started

1. Create a `.env` file:

```bash
cp .env.example .env
```

2. Define required environment variables in your `.env` file:

```env
# Azure Search Configuration
AZURE_AI_SEARCH_ENDPOINT=your_azure_search_endpoint
AZURE_AI_SEARCH_API_KEY=your_azure_search_api_key
AZURE_AI_SEARCH_INDEX_NAME=your_index_name
AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME=unit_lesson

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Arcade Tools Configuration
ARCADE_API_KEY=your_arcade_api_key
```

3. Install dependencies:

```bash
uv pip install -e .
```

4. Run tests:

```bash
python -m pytest tests/ -v
```

## Architecture

The project is structured around several key components:

- `src/agent_arcade_tools/graph.py`: Defines the main workflow using LangGraph
- `src/agent_arcade_tools/configuration.py`: Handles configuration and settings
- `src/agent_arcade_tools/tools.py`: Contains tool definitions and implementations
- `src/agent_arcade_tools/state.py`: Manages state throughout the workflow

## Development

The project uses:

- `uv` for Python package management
- `pytest` for testing
- `langgraph` for workflow management
- `langchain-arcade` for Arcade Tools integration
- Azure Search for semantic search capabilities

## Testing

The project includes both unit and integration tests:

- Unit tests verify individual components
- Integration tests ensure the entire workflow functions correctly

Run tests with:

```bash
python -m pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## TODO

- [ ] Implement server-side user_id validation
  - Add validation middleware
  - Ensure user_id is present and valid before processing requests
  - Add appropriate error handling for invalid user_ids

## License

This project is licensed under the MIT License - see the LICENSE file for details.
