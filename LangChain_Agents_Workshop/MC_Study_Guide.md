# Model Context Protocol (MCP): Study Guide

## Based on BEL C16 AI W6 Session 9 and 10 Deck

This guide is intentionally aligned to the MCP portion of the deck, especially slides 25-40. It stays focused on:

- What MCP is
- Why MCP exists
- How MCP works
- MCP vs APIs
- MCP client, host, server, tools, and resources
- Tool call structure
- Building one simple Python MCP server
- Connecting it to Claude Desktop and Cursor

---

## Suggested 2-Hour Flow

| Time | Topic | Goal |
|------|------|------|
| 0-15 min | What MCP is | Build the core definition |
| 15-30 min | Why MCP exists | Show the problem MCP solves |
| 30-50 min | MCP architecture | Explain host, client, server, tools, resources |
| 50-65 min | MCP vs API | Clarify confusion with analogies |
| 65-80 min | Tool-call flow | Show the JSON request/response idea |
| 80-110 min | Hands-on | Build SQLite MCP server + connect in Cursor/Claude |
| 110-120 min | Demo + recap | Reinforce the end-to-end picture |

---

## 1. What Is MCP?

**Model Context Protocol (MCP)** is a standard way for AI applications to talk to tools and data sources.

Simple definition:

> MCP is a common communication protocol between an AI app and external tools.

Instead of building a custom integration for every AI app and every tool, MCP gives us a shared format and shared flow.

### One-Line Mental Model

**API is for software-to-software communication. MCP is for model-to-tool communication.**

---

## 2. Why MCP Exists

Models today often need access to:

- Local files
- Editor commands
- Private databases
- Internal tools
- System actions

Traditional APIs alone are not enough for this use case because:

- APIs are usually designed for developers writing application code
- APIs do not automatically solve tool approval and permission flow for AI apps
- Every AI app might need a separate custom integration
- Local tools and private machine resources are harder to expose safely through regular APIs

### Core Problem

Without MCP:

- every AI tool integration is custom
- tools behave differently across apps
- safety and permissions are inconsistent
- switching AI clients becomes painful

With MCP:

- the interface is standardized
- tools become reusable
- clients can handle permissions
- the same server can work with multiple AI apps

---

## 3. MCP Architecture: Host, Client, Server, Tools, Resources

The deck names **client, server, tools, resources, messages, and model**. Many people also use the word **host**.

### Important Clarification

For teaching purposes:

- **Host** = the application where the model is running for the user
- **Client** = the MCP communication layer inside that app

So in practice, **Claude Desktop** or **Cursor** can be explained as the **host application**, and inside it an **MCP client** talks to the server.

### Components

#### 1. Host Application

Examples:

- Claude Desktop
- Cursor

This is what the learner actually opens and chats with.

#### 2. MCP Client

The part inside the host that:

- discovers available tools
- sends MCP requests
- receives responses
- manages permissions and approvals

#### 3. MCP Server

This is your tool host.

It exposes capabilities in a standardized way.  
In our session, the MCP server will expose SQLite data access tools.

#### 4. Tools

Tools are callable actions.

Examples:

- `list_tables`
- `monthly_revenue`
- `top_products`

Think of tools as functions with:

- a name
- input schema
- JSON output

#### 5. Resources

Resources are read-oriented pieces of data and are useful when data changes over time.

Examples from the deck:

- folder watchers
- log streams
- database updates
- realtime metrics

For today’s session, we will **mention resources clearly** but keep the hands-on focused on **tools**, since that is the simplest path for a first demo.

#### 6. Model

The model decides whether it needs a tool.

It does not directly access the database itself.  
It asks through the MCP flow.

---

## 4. End-to-End MCP Flow

Here is the full sequence in a simple diagram:

```text
User asks question
        |
        v
Claude Desktop / Cursor (Host App)
        |
        v
Model decides: "I need a tool"
        |
        v
MCP Client sends JSON request
        |
        v
MCP Server receives tool call
        |
        v
Tool runs (for example, query SQLite)
        |
        v
Server returns JSON result
        |
        v
Model reads result and continues reasoning
        |
        v
Final answer to user
```

### Short Version

**User -> Model -> Client -> Server -> Tool -> JSON result -> Model -> Answer**

---

## 5. Best Analogy to Explain MCP

### Analogy: Universal Waiter System

Imagine many customers want to order from many kitchens.

Without MCP:

- every customer must learn each kitchen’s custom ordering style
- every kitchen expects a different format
- there is no standard approval or workflow

With MCP:

- everyone uses the same order form
- the waiter knows how to send the request
- the kitchen only needs to understand one standard format

Mapping:

- **User** = customer
- **Host/Client** = waiter/front desk
- **MCP Server** = kitchen counter for a specific system
- **Tool** = a menu action
- **JSON result** = prepared response returned in a standard format

This analogy helps explain why the protocol matters more than the individual tool.

---

## 6. MCP vs API

This is usually the biggest confusion point for learners.

### Simple Difference

An **API** is a general interface for software systems.  
**MCP** is a standard specifically designed so AI applications can safely and consistently use tools.

### Comparison Table

| APIs | MCP |
|------|-----|
| General-purpose software integration | AI-to-tool integration standard |
| Often network-first | Can work with local or internal tools |
| Built for developers writing app logic | Built for AI clients and model tool use |
| No standard permission workflow for AI apps | Permission-aware client flow |
| Integration often differs per tool/provider | Standard tool interaction pattern |
| Models may struggle with inconsistent interfaces | Predictable JSON tool format |

### Easy Classroom Explanation

You can say:

> An API tells a developer how to call a service. MCP tells an AI application how to safely discover and use tools in a standard way.

### Another Analogy

- **API** is like a machine manual
- **MCP** is like a standard power socket

A manual explains one machine.  
A standard socket lets many devices work with the same plug style.

---

## 7. When Should We Use MCP?

Use MCP when:

- multiple AI apps need the same tool
- the tool is local
- the tool connects to private/internal data
- you want consistent permissions
- you want to switch clients without rewriting the integration

Good examples:

- local file search
- editor commands
- internal logs
- SQLite or internal database access
- private business tools

---

## 8. Tool Call Structure

From the deck, the basic structure is:

- `tool_name`
- `arguments`
- `result`
- `error`

### Example

```json
{
  "tool_name": "monthly_revenue",
  "arguments": {
    "year": 2025
  }
}
```

Possible result:

```json
{
  "rows": [
    {"month": "Jan", "revenue": 18200},
    {"month": "Feb", "revenue": 21500}
  ],
  "row_count": 2
}
```

Possible error:

```json
{
  "error": {
    "message": "Table sales not found",
    "type": "database_error"
  }
}
```

### Teaching Note

Repeat this idea clearly:

> The server should always return structured JSON. No silent failures. Clear error messages.

That directly matches the deck guidelines.

---

## 9. Tools vs Resources

This is another small but important teaching point.

### Tools

Use tools when the model needs to **do something**.

Examples:

- run a database query
- calculate a summary
- create a report

### Resources

Use resources when the model needs to **read or subscribe to data**.

Examples:

- a log stream
- a folder listing
- realtime metrics

### Short Rule

- **Tool = action**
- **Resource = accessible data/context**

For this workshop, our SQLite example is primarily **tool-based**.

---

## 10. Our Session Demo: SQLite Sales Database

We will use a very simple local database example because it makes MCP concrete.

### Why SQLite Is a Good Demo

- local and easy to explain
- no cloud dependency
- learners understand tables quickly
- easy to show structured JSON output
- easy to connect with data visualization in the notebook

### Demo Story

We create a small sales database with dummy data, then expose it through an MCP server with tools like:

- `list_tables`
- `show_sample_sales`
- `monthly_revenue`
- `top_products`

Then:

- Claude Desktop or Cursor can call these tools
- the model can answer questions using the returned data
- the notebook can query the same SQLite DB and plot charts with `matplotlib`

This keeps the learning grounded and visual.

---

## 11. Architecture Diagram for the SQLite Demo

```text
+---------------------------+
| User asks in Claude/Cursor|
+-------------+-------------+
              |
              v
+---------------------------+
| Host App + MCP Client     |
| (Claude Desktop / Cursor) |
+-------------+-------------+
              |
       JSON tool request
              |
              v
+---------------------------+
| Python MCP Server         |
| sqlite_sales_demo         |
+-------------+-------------+
              |
       runs tool function
              |
              v
+---------------------------+
| SQLite Database           |
| sales, products, regions  |
+-------------+-------------+
              |
       JSON tool result
              |
              v
+---------------------------+
| Model uses result         |
| and answers user          |
+---------------------------+
```

---

## 12. Slide-Aligned Teaching Script

### Slides 25-26: What learners will cover

Say:

> Today we are not learning every MCP feature. We are learning the core idea: what MCP is, why it exists, how the flow works, and how to build one simple server.

### Slides 27-31: What MCP is and why it exists

Say:

> MCP is a standard for AI-to-tool communication. The value is not just calling a tool. The value is calling tools in a standard, reusable, permission-aware way.

### Slides 32-35: Components and workflow

Say:

> The user chats with an AI app like Claude Desktop or Cursor. Inside that app, the model decides whether a tool is needed. The MCP client sends a request to the MCP server. The server executes the tool and returns JSON. The model then uses that result to continue.

### Slide 36: Tool structure

Say:

> Every tool interaction has a name, arguments, and a structured result. If something fails, return a clear structured error. This predictability is what makes MCP useful.

### Slides 38-40: Build steps and guidelines

Say:

> Our server will follow the same sequence: define a tool, define inputs, return JSON, register it, run the server, and test it in the client.

---

## 13. Hands-On Flow

### Part A: Create Dummy SQLite Data

- create a `sales` table
- insert sample rows
- query it directly in the notebook

Goal: learners see the data before MCP is introduced.

### Part B: Build the MCP Server

- define the server
- define 3-4 simple tools
- ensure each tool returns JSON

Goal: learners see that MCP is just exposing clean tool interfaces on top of useful code.

### Part C: Connect to Cursor

- add the server in `.cursor/mcp.json`
- restart/reload Cursor
- inspect available tools
- ask questions that trigger the tools

### Part D: Connect to Claude Desktop

- register the local server with Claude Desktop
- restart the app if needed
- confirm the tool list appears
- ask database questions

### Part E: Visualize in Notebook

- query the SQLite DB directly
- load into pandas
- create a simple bar chart or line chart using `matplotlib`

This closes the loop between:

- database
- MCP tool access
- analysis
- visualization

---

## 14. Demo Prompts You Can Use Live

In Cursor or Claude Desktop:

- "What tables are available in the sales database?"
- "Show the top 5 products by revenue."
- "Summarize monthly revenue trends."
- "Which region performed best?"
- "Use the SQLite sales tools to answer this question."

In the notebook:

- "Plot monthly revenue as a bar chart."
- "Create a product-wise revenue chart."
- "Compare region-wise totals."

---

## 15. Common Learner Confusions

### Confusion 1: Is MCP replacing APIs?

No.

MCP does not replace APIs.  
It gives AI apps a standard way to use tools, and those tools may themselves call APIs.

### Confusion 2: Is the model directly querying the database?

No.

The model requests a tool call.  
The MCP server runs the query and returns structured output.

### Confusion 3: Is Claude Desktop the server?

No.

Claude Desktop is the host app and contains the MCP client.  
Your Python program is the MCP server.

### Confusion 4: Why not directly write Python code in the app?

Because MCP makes tools reusable across clients and standardizes how they are exposed.

---

## 16. Best Practices from the Deck

- always return JSON
- keep field names consistent
- include metadata if useful
- use clear error messages
- never fail silently

You can turn this into a rule learners repeat:

> Good MCP tools are predictable, structured, and explicit.

---

## 17. Minimal Recap

If you want to end with a crisp summary, use this:

> MCP is a standard way for AI apps to use external tools and data.  
> The model decides when a tool is needed.  
> The client sends a structured request to the MCP server.  
> The server runs the tool and returns JSON.  
> This makes integrations more reusable, safer, and easier to carry across AI clients like Claude Desktop and Cursor.

---

## 18. What We Will Build

By the end of the session, learners should have seen:

- one local SQLite database
- one Python MCP server
- a few simple tools
- one MCP-aware client connection
- one visualization notebook

That is enough to give them a strong first mental model of MCP without going beyond the deck.
