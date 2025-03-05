FROM langchain/langgraph-api:3.12

# Copy the entire project
COPY . /app
WORKDIR /app

# Install the project in editable mode
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e .

# Set the graph configuration to use the installed package
ENV LANGSERVE_GRAPHS='{"graph": "agent_arcade_tools.graph:graph"}'

