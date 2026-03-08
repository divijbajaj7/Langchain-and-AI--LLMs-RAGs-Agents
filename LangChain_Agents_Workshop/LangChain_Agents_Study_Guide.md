# LangChain Agents: A Comprehensive Study Guide

## When LLMs Decide What To Do Next

---

## Table of Contents
1. [Introduction: From Chains to Agents](#1-introduction-from-chains-to-agents)
2. [Understanding Agents](#2-understanding-agents)
3. [When to Use Agents vs Chains](#3-when-to-use-agents-vs-chains)
4. [Core Components of a LangChain Agent](#4-core-components-of-a-langchain-agent)
5. [Deep Dive: Tools](#5-deep-dive-tools)
6. [Agent Types Explained](#6-agent-types-explained)
7. [Adding Memory to Agents](#7-adding-memory-to-agents)
8. [Guardrails and Controls](#8-guardrails-and-controls)
9. [Debugging Agents](#9-debugging-agents)
10. [Best Practices Summary](#10-best-practices-summary)

---

## 1. Introduction: From Chains to Agents

### Quick Recap: What are Chains?

In previous sessions, you learned about **chains** - the fundamental building blocks in LangChain that connect multiple components (prompts, LLMs, parsers) into a pipeline.

```
Input → Step 1 → Step 2 → Step 3 → Output
```

**Key characteristics of chains:**
- Follow a **fixed, predefined flow**
- **You** define the sequence of operations
- Each step is deterministic
- Great when you know exactly what needs to happen

**Example Chain Flow:**
```
User Question → Retrieve Documents → Generate Answer → Format Output
```

### What Makes Agents Different?

**Agents** represent a paradigm shift: instead of you defining every step, the **LLM decides what to do next**.

```
Goal → [LLM Thinks] → Tool Call → Observation → [LLM Thinks Again] → Answer
        ↑                                              |
        └──────────────────────────────────────────────┘
                    (Loop until done)
```

**Key insight:** Agents are useful when you don't know the exact path to the answer at the start.

---

## 2. Understanding Agents

### Definition

> An **agent** is a system where an LLM is given a goal and a set of tools, then autonomously decides which tools to use and in what order to achieve that goal.

### The Agent Loop (Mental Model)

Every agent follows this fundamental loop:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    1. GOAL           User provides a task/question          │
│         ↓                                                   │
│    2. THINK          LLM analyzes goal + current state      │
│         ↓                                                   │
│    3. ACT            LLM chooses a tool and arguments       │
│         ↓                                                   │
│    4. OBSERVE        Tool executes, returns result          │
│         ↓                                                   │
│    5. UPDATE         Result added to scratchpad             │
│         ↓                                                   │
│    6. REPEAT         Go back to THINK (or ANSWER if done)   │
│         ↓                                                   │
│    7. FINAL ANSWER   LLM provides the response              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The Scratchpad Concept

The **scratchpad** is the agent's working memory during a single task. It accumulates:
- The original goal
- Each action taken
- Each observation received
- Reasoning at each step

Think of it as the agent "showing its work" - like scratch paper during a math exam.

**Example Scratchpad:**
```
Goal: What's the weather in Tokyo and convert the temperature to Fahrenheit?

Thought: I need to first get the weather in Tokyo
Action: get_weather
Action Input: {"city": "Tokyo"}
Observation: Temperature: 15°C, Condition: Cloudy

Thought: Now I need to convert 15°C to Fahrenheit
Action: calculator
Action Input: {"expression": "(15 * 9/5) + 32"}
Observation: 59

Thought: I now have all the information needed
Final Answer: The weather in Tokyo is cloudy with a temperature of 15°C (59°F).
```

---

## 3. When to Use Agents vs Chains

### Use an Agent When:

| Scenario | Why Agent? |
|----------|-----------|
| System must **choose actions based on context** | Path depends on intermediate results |
| Tasks involve **research, troubleshooting, or planning** | Multiple data sources, unknown steps |
| Path is **not obvious** before you start | Can't pre-define the sequence |
| User queries are **open-ended** | "Help me understand X" type questions |

**Real-world agent use cases:**
- Customer support bots that check orders, update info, escalate issues
- Research assistants that search, summarize, and cite sources
- DevOps agents that check logs, metrics, and create tickets
- Data analysis agents that query databases based on findings

### Avoid Agents When:

| Scenario | Why Not Agent? |
|----------|---------------|
| Flow is **simple and fixed** | Chain is simpler and more reliable |
| You need **predictable latency and cost** | Agents can loop unpredictably |
| Output format must be **strict and repeatable** | Agents have variable paths |
| Task is **well-defined** | No need for dynamic decision-making |

**Use chains instead for:**
- Form submission → Email notification
- Document → Summarization
- Translation pipelines
- Fixed Q&A with known retrieval steps

### Visual Comparison

```
CHAIN:                              AGENT:
┌─────┐   ┌────┐   ┌────┐          ┌────────┐
│Input│ → │Step│ → │Step│ → Output │ Metrics│ ←─┐
└─────┘   │ 1  │   │ 2  │          └────────┘   │
          └────┘   └────┘               ↑       │
                                    ┌───────┐   │
                                    │ Agent │───┤
                                    └───────┘   │
                                        ↓       │
                                    ┌───────┐   │
                                    │ Logs  │ ──┘
                                    └───────┘
```

---

## 4. Core Components of a LangChain Agent

An agent in LangChain consists of four main pieces:

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────────┐   ┌──────┐
│  Tools  │ → │   LLM   │ → │  Agent  │ → │ AgentExecutor   │ → │ User │
└─────────┘   └─────────┘   │  Config │   └─────────────────┘   └──────┘
                            └─────────┘
```

### 4.1 LLM / Chat Model (The Brain)

The LLM is the reasoning engine that:
- Understands the user's goal
- Decides which tool to use
- Formulates tool arguments
- Interprets tool outputs
- Determines when to stop

**Requirements for an agent LLM:**
- Should support tool/function calling (GPT-4, Claude, etc.)
- Good instruction-following capability
- Ability to reason about multi-step problems

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0  # Lower temperature = more deterministic tool selection
)
```

### 4.2 Tools (Functions the Agent Can Call)

Tools are the agent's hands - functions it can execute to interact with the world.

Each tool has:
- **Name**: How the agent refers to it
- **Description**: What the tool does (crucial for tool selection!)
- **Input Schema**: What arguments it accepts
- **Function**: The actual code that runs

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any math calculations."""
    return str(eval(expression))
```

### 4.3 Agent Type (The Strategy)

The agent type determines HOW the LLM decides to use tools:

| Agent Type | How It Works | Best For |
|------------|-------------|----------|
| **Tool Calling Agent** | Uses native function calling API | Modern models (GPT-4, Claude) |
| **ReAct Agent** | Explicit Thought → Action → Observation | Debugging, transparency |
| **Structured Chat** | JSON-based tool selection | Models without function calling |

### 4.4 AgentExecutor (The Control Loop)

The `AgentExecutor` is the orchestrator that:
- Connects all pieces together
- Runs the think-act-observe loop
- Handles tool execution
- Manages stopping conditions
- Applies guardrails

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10
)
```

---

## 5. Deep Dive: Tools

### Anatomy of a Tool

Every tool in LangChain has four components:

```
┌─────────────────────────────────────────────────────┐
│                       TOOL                          │
├─────────────────────────────────────────────────────┤
│  name: "search_logs"                                │
│                                                     │
│  description: "Search application logs by service  │
│               name and time range. Use this when   │
│               you need to find error messages or   │
│               debug issues in a specific service." │
│                                                     │
│  args_schema: {                                     │
│      service_name: str (required)                   │
│      from_time: str (optional)                      │
│      to_time: str (optional)                        │
│  }                                                  │
│                                                     │
│  function: def search_logs(service_name, ...): ... │
└─────────────────────────────────────────────────────┘
```

### Why Descriptions Matter

**Critical insight:** The LLM only sees the tool's name and description when deciding what to use. It never sees your code!

**Bad description:**
```python
@tool
def search(query: str) -> str:
    """Search function."""  # Too vague! Agent won't know when to use this
```

**Good description:**
```python
@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic.
    Use this when you need up-to-date information that might not
    be in your training data, such as recent news, current prices,
    or live data. Returns a summary of relevant web pages."""
```

### Tool Design Principles

| DO | DON'T |
|----|-------|
| Use **action-style names**: `search_logs`, `create_ticket`, `get_weather` | Use vague names: `process`, `handle`, `do_thing` |
| Define **clear arguments**: `service_name`, `time_range`, `severity` | Use complex nested objects |
| Return **simple formats**: text or basic JSON | Return unpredictable structures |
| Make tools **single-purpose** | Mix multiple jobs in one tool |
| Include **usage guidance** in description | Assume the model knows when to use it |

### Creating Custom Tools

**Method 1: @tool decorator (simplest)**
```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b
```

**Method 2: StructuredTool (more control)**
```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="The city to get weather for")
    units: str = Field(default="celsius", description="Temperature units: celsius or fahrenheit")

def get_weather(city: str, units: str = "celsius") -> str:
    # Implementation here
    return f"Weather in {city}: 22°{units[0].upper()}"

weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="get_weather",
    description="Get current weather for a city",
    args_schema=WeatherInput
)
```

---

## 6. Agent Types Explained

### 6.1 Tool Calling Agent (Recommended)

Uses the model's native function/tool calling capability.

**How it works:**
1. Tools are passed as function definitions to the API
2. Model outputs a structured tool call (name + arguments)
3. No parsing needed - structured output from the model

**Best for:** Modern models (GPT-4, Claude 3, etc.)

```python
from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)
```

### 6.2 ReAct Agent (Reasoning + Acting)

Explicitly generates reasoning before each action.

**How it works:**
1. Model generates: "Thought: I need to..."
2. Then: "Action: tool_name"
3. Then: "Action Input: {args}"
4. System returns: "Observation: result"
5. Repeat until "Final Answer"

**Best for:** Debugging, transparency, models without function calling

```python
from langchain.agents import create_react_agent
from langchain import hub

# Get the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
```

**Example ReAct trace:**
```
Thought: The user wants to know the weather in Paris and convert it to Fahrenheit.
         I should first get the weather.
Action: get_weather
Action Input: {"city": "Paris"}
Observation: Temperature: 18°C, Condition: Sunny

Thought: Now I need to convert 18°C to Fahrenheit using the calculator.
Action: calculator
Action Input: {"expression": "(18 * 9/5) + 32"}
Observation: 64.4

Thought: I now have all the information to answer.
Final Answer: The weather in Paris is sunny with a temperature of 18°C (64.4°F).
```

### 6.3 Structured Chat Agent

Uses JSON formatting for tool selection.

**Best for:** Models that don't support native function calling but can output JSON

### Comparison

| Feature | Tool Calling | ReAct | Structured Chat |
|---------|-------------|-------|-----------------|
| Reliability | High | Medium | Medium |
| Transparency | Low | High | Medium |
| Model Requirements | Function calling API | Any chat model | JSON output |
| Debugging | Harder | Easy | Medium |
| Speed | Fast | Slower (more tokens) | Medium |

---

## 7. Adding Memory to Agents

### Why Memory Helps

Without memory, agents are **stateless** - each query starts fresh with no knowledge of previous interactions.

**Problems without memory:**
- "What was that city I asked about?" → Agent doesn't know
- Multi-step tasks fail if they span multiple calls
- No continuity in conversations

### Integrating Memory with Agents

**Step 1: Create memory**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

**Step 2: Include chat_history in prompt**
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    MessagesPlaceholder(variable_name="chat_history"),  # Inject history here
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
```

**Step 3: Add memory to AgentExecutor**
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)
```

### Memory Types for Agents

| Memory Type | Use Case | Trade-off |
|-------------|----------|-----------|
| **ConversationBufferMemory** | Short conversations | Full history, can grow large |
| **ConversationBufferWindowMemory** | Longer chats | Only last K turns |
| **ConversationSummaryMemory** | Very long sessions | Summarizes history, loses detail |

---

## 8. Guardrails and Controls

### Why Guardrails Matter

Agents can:
- Loop indefinitely (costly!)
- Call wrong tools repeatedly
- Generate unpredictable outputs
- Take too long to respond

### Available Controls

**1. Limit Iterations**
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,  # Stop after 5 tool calls
    max_execution_time=30  # Stop after 30 seconds
)
```

**2. Handle Parsing Errors**
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True  # Don't crash on malformed output
)
```

**3. Early Stopping**
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    early_stopping_method="generate"  # Generate final answer if max reached
)
```

**4. Restrict Tool Access**
```python
# Different tool sets for different users/tasks
admin_tools = [search, delete, modify]
user_tools = [search]  # Limited access

agent_executor = AgentExecutor(
    agent=agent,
    tools=user_tools  # Only these tools available
)
```

### Input/Output Validation

```python
from pydantic import BaseModel, validator

class SafeInput(BaseModel):
    query: str

    @validator('query')
    def no_injection(cls, v):
        forbidden = ['DROP', 'DELETE', ';--']
        if any(f in v.upper() for f in forbidden):
            raise ValueError("Potentially unsafe input")
        return v
```

---

## 9. Debugging Agents

### Enable Verbose Logging

The single most important debugging tool:

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True  # See everything!
)
```

**Verbose output shows:**
- Each thought/reasoning step
- Tool selected
- Arguments passed
- Tool output received
- Final answer generation

### What to Look For

**Problem: Agent calls wrong tool**
→ Check tool descriptions - are they clear and distinct?

**Problem: Agent loops forever**
→ Add `max_iterations`, check if tool returns useful info

**Problem: Agent gives up too early**
→ Check if tool output is too long/confusing

**Problem: Agent hallucinates tool names**
→ Verify tools are properly registered, check tool names

### Debugging Checklist

1. **Enable verbose mode** - always start here
2. **Log tool calls** - name, input, output
3. **Review thought traces** - for ReAct agents
4. **Reproduce with small prompts** - isolate the issue
5. **Check tool descriptions** - most common fix!
6. **Verify tool outputs** - are they what the model expects?

### Common Fixes

| Issue | Fix |
|-------|-----|
| Wrong tool selection | Improve tool description, add examples |
| Infinite loops | Add max_iterations, improve stopping criteria |
| Bad arguments | Add argument descriptions in schema |
| Slow performance | Reduce tool count, simplify outputs |
| Parsing errors | Enable handle_parsing_errors |

---

## 10. Best Practices Summary

### Tool Design
- Keep **actions small and clear** - one tool, one job
- Use **as few tools as possible** - more tools = more confusion
- **Test tool descriptions** - try describing to a colleague
- Return **simple, predictable formats**

### Agent Configuration
- **Limit agent freedom** - use max_iterations, timeouts
- Start with **low max_iterations** and increase if needed
- Use **verbose=True** during development
- Choose the **right agent type** for your model

### Development Process
- **Log everything** in development
- Start with **offline test prompts** before going live
- **Test edge cases** - what if tool fails? What if no result?
- **Monitor in production** - track iterations, latency, cost

### When Things Go Wrong

```
┌─────────────────────────────────────────────────────────────┐
│                    DEBUGGING FLOWCHART                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Agent not working?                                         │
│         │                                                   │
│         ▼                                                   │
│  Enable verbose=True                                        │
│         │                                                   │
│         ▼                                                   │
│  Is the right tool being selected?                          │
│         │                                                   │
│    NO ──┼── YES                                             │
│    │    │                                                   │
│    │    ▼                                                   │
│    │  Are arguments correct?                                │
│    │         │                                              │
│    │    NO ──┼── YES                                        │
│    │    │    │                                              │
│    │    │    ▼                                              │
│    │    │  Is tool output useful?                           │
│    │    │         │                                         │
│    │    │    NO ──┼── YES                                   │
│    │    │    │    │                                         │
│    │    │    │    ▼                                         │
│    │    │    │  Check final answer logic                    │
│    │    │    │                                              │
│    ▼    ▼    ▼                                              │
│  Fix    Fix  Fix                                            │
│  tool   arg  tool                                           │
│  desc   desc output                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

### Minimal Agent Setup
```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. Tools
@tool
def calculator(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression))

tools = [calculator]

# 3. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 4. Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Run
result = agent_executor.invoke({"input": "What is 25 * 4?"})
```

---

## Key Takeaways

1. **Agents vs Chains**: Chains = fixed path, Agents = dynamic decisions
2. **Agent Loop**: Goal → Think → Act → Observe → Repeat → Answer
3. **Tools are crucial**: Good descriptions = good tool selection
4. **Start simple**: Few tools, low iterations, verbose logging
5. **Debug with verbose**: Always your first step
6. **Add guardrails**: Protect against loops and runaway costs
7. **Memory enables continuity**: Essential for multi-turn conversations

---

*This guide accompanies the LangChain Agents Workshop notebook. Practice the concepts hands-on to solidify your understanding!*
