I'll help write clear instructions for the developer to understand and fix the issues. Here's a comprehensive set of instructions:

# Developer Instructions: LangGraph Workflow Fix

## Overview
The LangGraph workflow is experiencing issues with state management and message handling, particularly around the agent node and tool execution. The main problems are:

1. Agent node hanging
2. Incorrect state transitions
3. Message format inconsistencies
4. Tool execution flow issues

## Current Architecture
- `StateGraph` with three main nodes: agent, authorization, and tools
- Message state management using `MessagesState`
- Tool execution through `ToolNode`
- Authorization handling for protected tools

## Key Issues to Address

### 1. Message Processing in `call_agent`
```python
def call_agent(state: MessagesState) -> MessagesState:
```
- Current issue: Message format conversion is causing state inconsistencies
- Required changes:
  - Preserve original message types
  - Handle tool messages correctly
  - Ensure proper message state updates

### 2. Tool Execution Flow
```python
def handle_tools(state: MessagesState) -> dict[str, Sequence[BaseMessage]]:
```
- Current issue: Tool call conversion and execution is not maintaining state correctly
- Required changes:
  - Use `json.loads` instead of `eval` for arguments
  - Maintain consistent tool call types
  - Properly handle tool responses

### 3. State Transitions
```python
def should_continue(state: MessagesState) -> str:
```
- Current issue: Incorrect node transitions
- Required changes:
  - Properly handle tool message transitions
  - Correct authorization flow
  - Maintain state consistency

## Test Cases to Fix
1. `test_message_history_maintenance`
2. `test_ai_message_tool_calls_validation`
3. `test_authorized_complete_flow`
4. `test_email_check_flow`
5. `test_email_check_flow_with_exact_payload`
6. `test_agent_to_authorization_transition`
7. `test_auth_to_tools_error_transition`
8. `test_should_continue_with_tool_message`
9. `test_agent_response_to_tool_message`

## Implementation Guidelines

### 1. Message State Management
- Always use `MessagesState` for state updates
- Preserve message types and metadata
- Handle tool messages consistently

### 2. Tool Execution
- Use `ToolNode` for tool execution
- Convert tool calls properly
- Maintain tool call IDs and status

### 3. Authorization Flow
- Check authorization requirements first
- Handle interrupts properly
- Maintain state during authorization

### 4. Graph Configuration
```python
workflow = StateGraph(MessagesState, AgentConfigurable)
workflow.add_node("agent", call_agent)
workflow.add_node("authorization", authorize)
workflow.add_node("tools", handle_tools)
```

## Testing Instructions
1. Run integration tests:
```bash
uv run pytest tests/integration_tests/test_graph.py -v
```

2. Verify each test case:
   - Message history maintenance
   - Tool call validation
   - Authorization flow
   - Email check flow
   - State transitions

## Dependencies
- LangGraph SDK
- LangChain Core
- Arcade Tool Manager
- OpenAI API

## Environment Setup
1. Required environment variables:
   - `ARCADE_API_KEY`
   - `OPENAI_API_KEY`

2. Package management:
   - Use `uv` for Python dependencies
   - Install dev dependencies: `uv pip install -e ".[dev]"`

## Success Criteria
1. All test cases pass
2. No state inconsistencies
3. Proper message handling
4. Correct tool execution
5. Smooth authorization flow

## Additional Notes
- Follow LangGraph SDK best practices
- Maintain proper logging
- Handle errors gracefully
- Document any changes
- Keep the code clean and maintainable

Would you like me to provide more specific details about any of these areas?
