# LangChain Components: A Visual Study Guide

## Prompts, Parsers, Memory & Chains - Deep Dive

---

## Table of Contents
1. [The Big Picture: How Components Connect](#1-the-big-picture)
2. [Prompt Templates](#2-prompt-templates)
3. [Output Parsers](#3-output-parsers)
4. [Memory Systems](#4-memory-systems)
5. [Chains](#5-chains)
6. [Putting It All Together](#6-putting-it-all-together)
7. [Quick Reference](#7-quick-reference)

---

## 1. The Big Picture

### Why LangChain?

LLMs are **stateless** - they don't remember anything between calls. Every API call is independent:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WITHOUT LANGCHAIN                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  You: "My name is Alice"                                            │
│  LLM: "Nice to meet you, Alice!"                                    │
│                                                                     │
│  You: "What's my name?"                                             │
│  LLM: "I don't know your name"  ← FORGOT! Each call is isolated    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

LangChain solves this and more by providing **building blocks** that work together:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      THE LANGCHAIN ECOSYSTEM                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌─────┐    ┌────────┐    ┌────────────┐          │
│   │  PROMPT  │ →  │ LLM │ →  │ PARSER │ →  │  MEMORY    │          │
│   │ TEMPLATE │    │     │    │        │    │ (optional) │          │
│   └──────────┘    └─────┘    └────────┘    └────────────┘          │
│        ↑                                          │                 │
│        │                                          │                 │
│        └──────────── CHAIN ──────────────────────┘                 │
│              (connects everything)                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | What It Does | Why You Need It |
|-----------|-------------|-----------------|
| **Prompt Template** | Creates structured prompts with variables | Reusability, consistency, better outputs |
| **Output Parser** | Converts LLM text → structured data | Get lists, JSON, objects instead of strings |
| **Memory** | Stores conversation history | Multi-turn conversations, context awareness |
| **Chain** | Connects components together | Build complex workflows, automate pipelines |

---

## 2. Prompt Templates

### The Problem: Why Not Just Use Strings?

```python
# ❌ BAD: Hardcoded string
prompt = "Explain artificial intelligence to a beginner"

# ❌ BAD: Manual string formatting (error-prone)
prompt = f"Explain {topic} to a {audience}"  # What if topic is None?

# ✅ GOOD: Prompt Template
template = PromptTemplate(template="Explain {topic} to a {audience}")
prompt = template.format(topic="AI", audience="beginner")
```

### Visual: How Prompt Templates Work

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PROMPT TEMPLATE FLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TEMPLATE                  VARIABLES                 FINAL PROMPT   │
│  ┌─────────────────┐      ┌───────────┐      ┌────────────────────┐│
│  │ "Explain {topic}│  +   │topic="AI" │  =   │"Explain AI to a   ││
│  │  to a {audience}"│     │audience=  │      │ beginner"          ││
│  │                 │      │"beginner" │      │                    ││
│  └─────────────────┘      └───────────┘      └────────────────────┘│
│                                                                     │
│  Template is REUSABLE - same template, different variables!         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Types of Prompt Templates

#### 2.1 PromptTemplate (Basic)

The simplest form - just a string with `{variables}`:

```python
from langchain_core.prompts import PromptTemplate

# Method 1: Automatic variable detection
template = PromptTemplate.from_template(
    "Summarize this topic: {topic}"
)

# Method 2: Explicit variables (more control)
template = PromptTemplate(
    template="Summarize {topic} in {style} style",
    input_variables=["topic", "style"]
)

# Usage
prompt = template.format(topic="Machine Learning", style="simple")
# Output: "Summarize Machine Learning in simple style"
```

**Visual representation:**

```
┌────────────────────────────────────────┐
│          PromptTemplate                │
├────────────────────────────────────────┤
│  template: "Summarize {topic} in       │
│            {style} style"              │
│                                        │
│  input_variables: ["topic", "style"]   │
├────────────────────────────────────────┤
│           ↓ .format()                  │
├────────────────────────────────────────┤
│  "Summarize Machine Learning in        │
│   simple style"                        │
└────────────────────────────────────────┘
```

#### 2.2 ChatPromptTemplate (For Chat Models)

Modern LLMs are **chat models** that understand roles (system, user, assistant):

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CHAT MESSAGE ROLES                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ SYSTEM MESSAGE                                               │   │
│  │ "You are a helpful finance expert. Be concise."              │   │
│  │                                                              │   │
│  │ → Sets personality, rules, constraints                       │   │
│  │ → Model follows these instructions                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ HUMAN MESSAGE (User)                                         │   │
│  │ "What is compound interest?"                                 │   │
│  │                                                              │   │
│  │ → The user's question or request                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ AI MESSAGE (Assistant)                                       │   │
│  │ "Compound interest is interest calculated on..."             │   │
│  │                                                              │   │
│  │ → The model's response                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Code example:**

```python
from langchain_core.prompts import ChatPromptTemplate

# Create chat template with roles
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Always respond in {tone} tone."),
    ("human", "{question}")
])

# Format with variables
messages = chat_template.format_messages(
    role="Python expert",
    tone="friendly",
    question="How do I use list comprehensions?"
)

# Result: List of message objects ready for the LLM
```

**Why use ChatPromptTemplate?**

| PromptTemplate | ChatPromptTemplate |
|----------------|-------------------|
| Single string | Multiple messages with roles |
| Good for completion models | Good for chat models (GPT-4, Claude) |
| No role separation | System/Human/AI roles |
| Simple use cases | Complex conversations |

#### 2.3 FewShotPromptTemplate (Teaching by Example)

The most powerful technique: **show the model examples** of what you want:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEW-SHOT LEARNING                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  "I'll show you examples, then you do the same for my input"        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  PREFIX: "Give the opposite of each word."                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  EXAMPLE 1: Input: happy    →  Output: sad                   │   │
│  │  EXAMPLE 2: Input: tall     →  Output: short                 │   │
│  │  EXAMPLE 3: Input: fast     →  Output: slow                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  SUFFIX: "Input: bright"                                     │   │
│  │                                                              │   │
│  │  MODEL COMPLETES: "Output: dim"  ← Learned the pattern!      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Code example:**

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# Define examples
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "fast", "output": "slow"},
]

# Template for each example
example_template = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"]
)

# Few-shot prompt
few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the opposite of each word.",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)

# Use it
prompt = few_shot.format(word="bright")
# Model learns the pattern and outputs: "dim"
```

### Prompt Template Decision Tree

```
                    ┌─────────────────────┐
                    │ What type of prompt │
                    │    do you need?     │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │Simple string  │  │Chat model     │  │Model needs    │
    │with variables │  │with roles     │  │to learn       │
    └───────┬───────┘  └───────┬───────┘  │from examples  │
            │                  │          └───────┬───────┘
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │PromptTemplate │  │ChatPrompt-    │  │FewShotPrompt- │
    │               │  │Template       │  │Template       │
    └───────────────┘  └───────────────┘  └───────────────┘
```

---

## 3. Output Parsers

### The Problem: LLMs Return Strings!

```python
# LLM always returns text
response = llm.invoke("List 5 programming languages")
# Returns: "Here are 5 programming languages:\n1. Python\n2. Java..."

# But you wanted a Python list!
# How do you convert "Python, Java, C++..." → ["Python", "Java", "C++"]?
```

### Visual: Parser Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OUTPUT PARSER FLOW                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐   │
│  │  Prompt  │ →   │   LLM    │ →   │  Parser  │ →   │ Usable   │   │
│  │          │     │          │     │          │     │ Data     │   │
│  └──────────┘     └──────────┘     └──────────┘     └──────────┘   │
│                                                                     │
│                        │                   │                        │
│                        ▼                   ▼                        │
│                   "Python,            ["Python",                    │
│                    Java,               "Java",                      │
│                    C++"                "C++"]                       │
│                                                                     │
│                   (string)             (list)                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Types of Output Parsers

#### 3.1 StrOutputParser (Extract Text)

The simplest parser - extracts just the text content:

```
┌────────────────────────────────────────────────────────────────┐
│                   StrOutputParser                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  INPUT (AIMessage object):                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ AIMessage(                                                │ │
│  │   content="Hello, world!",                                │ │
│  │   response_metadata={...},                                │ │
│  │   token_usage={...}                                       │ │
│  │ )                                                         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ↓                                     │
│  OUTPUT (string):                                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ "Hello, world!"                                           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Why? Extract just the text, discard metadata                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

```python
from langchain_core.output_parsers import StrOutputParser

# In a chain
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "Python"})
# result is now a clean string, not AIMessage object
```

#### 3.2 CommaSeparatedListOutputParser (String → List)

```
┌────────────────────────────────────────────────────────────────┐
│             CommaSeparatedListOutputParser                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  INPUT (string):                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ "Python, Java, C++, JavaScript, Ruby"                     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ↓                                     │
│  OUTPUT (Python list):                                         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ ["Python", "Java", "C++", "JavaScript", "Ruby"]           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Now you can: iterate, filter, sort, count!                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

list_parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate(
    template="List 5 {category}. ONLY return a comma-separated list.",
    input_variables=["category"]
)

chain = prompt | llm | list_parser
result = chain.invoke({"category": "programming languages"})
# result = ["Python", "Java", "C++", "JavaScript", "Ruby"]

# Now you can work with it as a list!
for lang in result:
    print(f"- {lang}")
```

#### 3.3 JsonOutputParser (String → Dictionary)

For when you need structured data:

```
┌────────────────────────────────────────────────────────────────┐
│                   JsonOutputParser                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  INPUT (string with JSON):                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ '{                                                        │ │
│  │   "title": "Dune",                                        │ │
│  │   "author": "Frank Herbert",                              │ │
│  │   "year": 1965,                                           │ │
│  │   "genres": ["Sci-Fi", "Adventure"]                       │ │
│  │ }'                                                        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ↓                                     │
│  OUTPUT (Python dict):                                         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ {                                                         │ │
│  │   "title": "Dune",                                        │ │
│  │   "author": "Frank Herbert",                              │ │
│  │   "year": 1965,                                           │ │
│  │   "genres": ["Sci-Fi", "Adventure"]                       │ │
│  │ }                                                         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Access: result["title"], result["year"], etc.                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Define your data structure
class Book(BaseModel):
    title: str = Field(description="The title of the book")
    author: str = Field(description="The author of the book")
    year: int = Field(description="Publication year")
    genres: list = Field(description="List of genres")

json_parser = JsonOutputParser(pydantic_object=Book)

# The parser provides format instructions for the LLM!
prompt = PromptTemplate(
    template="Generate info about a {genre} book.\n{format_instructions}",
    input_variables=["genre"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()}
)

chain = prompt | llm | json_parser
result = chain.invoke({"genre": "science fiction"})
# result = {"title": "Dune", "author": "Frank Herbert", ...}
```

#### 3.4 PydanticOutputParser (String → Python Object)

The most powerful - returns actual Python objects with validation:

```
┌────────────────────────────────────────────────────────────────┐
│                  PydanticOutputParser                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  INPUT (string with JSON):                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ '{"movie_title": "The Matrix", "rating": 9, ...}'         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ↓                                     │
│  OUTPUT (Pydantic Object):                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ MovieReview(                                              │ │
│  │   movie_title="The Matrix",                               │ │
│  │   rating=9,                                               │ │
│  │   summary="Mind-bending sci-fi...",                       │ │
│  │   recommended=True                                        │ │
│  │ )                                                         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Access with dot notation: result.rating, result.movie_title   │
│  Validation: Pydantic checks types automatically!              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    movie_title: str = Field(description="Title of the movie")
    rating: int = Field(description="Rating from 1-10")
    summary: str = Field(description="Brief summary")
    recommended: bool = Field(description="Would you recommend?")

parser = PydanticOutputParser(pydantic_object=MovieReview)

# ... use in chain

result = chain.invoke({"movie": "The Matrix"})

# Access like an object!
print(result.movie_title)  # "The Matrix"
print(result.rating)       # 9
print(result.recommended)  # True
```

### Parser Comparison Chart

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE WHICH PARSER                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Need just text?                                                    │
│  └─→ StrOutputParser        "Hello world" → "Hello world"           │
│                                                                     │
│  Need a list?                                                       │
│  └─→ CommaSeparatedList     "a, b, c" → ["a", "b", "c"]             │
│                                                                     │
│  Need a dictionary?                                                 │
│  └─→ JsonOutputParser       "{...}" → {"key": "value"}              │
│                                                                     │
│  Need a typed object with validation?                               │
│  └─→ PydanticOutputParser   "{...}" → MyClass(attr="value")         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| Parser | Output Type | Use Case |
|--------|-------------|----------|
| `StrOutputParser` | `str` | Just need the text content |
| `CommaSeparatedListOutputParser` | `list` | Need items as a list |
| `JsonOutputParser` | `dict` | Need structured data, flexible |
| `PydanticOutputParser` | `Pydantic object` | Need typed objects, validation |

---

## 4. Memory Systems

### The Core Problem: LLMs Have No Memory

Every LLM call is **independent** - the model doesn't remember previous calls:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE MEMORY PROBLEM                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CALL 1:                         CALL 2:                            │
│  ┌────────────────────────┐     ┌────────────────────────┐         │
│  │ User: "I'm Alice"      │     │ User: "What's my name?"│         │
│  │ AI: "Hi Alice!"        │     │ AI: "I don't know"     │←─ Forgot!│
│  └────────────────────────┘     └────────────────────────┘         │
│                                                                     │
│  Each API call starts FRESH - no connection between calls!          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The Solution: Memory Injects History

Memory **stores** past conversations and **injects** them into new prompts:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HOW MEMORY WORKS                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                        MEMORY STORE                           │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │ Turn 1: User="I'm Alice" → AI="Hi Alice!"               │  │  │
│  │  │ Turn 2: User="I love pizza" → AI="Great choice!"        │  │  │
│  │  │ Turn 3: User="What's my name?" → ...                    │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     PROMPT (with history)                     │  │
│  │                                                               │  │
│  │  System: You are a helpful assistant.                         │  │
│  │                                                               │  │
│  │  ┌──────────────────────────────────────────────────────┐    │  │
│  │  │ HISTORY (injected from memory):                       │    │  │
│  │  │ Human: I'm Alice                                      │    │  │
│  │  │ AI: Hi Alice!                                         │    │  │
│  │  │ Human: I love pizza                                   │    │  │
│  │  │ AI: Great choice!                                     │    │  │
│  │  └──────────────────────────────────────────────────────┘    │  │
│  │                                                               │  │
│  │  Human: What's my name?                                       │  │
│  │                                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│                     AI: "Your name is Alice!"                       │
│                     (Now it remembers!)                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Types of Memory

#### 4.1 ConversationBufferMemory

Stores **EVERYTHING** - complete conversation history:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  ConversationBufferMemory                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conversation:                Memory stores:                        │
│  ┌───────────────────┐       ┌───────────────────────────────────┐ │
│  │ Turn 1            │   →   │ Human: "Hi, I'm Alice"            │ │
│  │ Turn 2            │   →   │ AI: "Hello, Alice!"               │ │
│  │ Turn 3            │   →   │ Human: "What's 2+2?"              │ │
│  │ Turn 4            │   →   │ AI: "4"                           │ │
│  │ Turn 5            │   →   │ Human: "I live in Paris"          │ │
│  │ Turn 6            │   →   │ AI: "Paris is beautiful!"         │ │
│  │ ...               │   →   │ ...                               │ │
│  │ Turn 100          │   →   │ (EVERYTHING is kept!)             │ │
│  └───────────────────┘       └───────────────────────────────────┘ │
│                                                                     │
│  ✅ PRO: Perfect recall - remembers everything                      │
│  ❌ CON: Grows unbounded - can exceed token limits!                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
# Stores complete history - simple but can grow large
```

#### 4.2 ConversationBufferWindowMemory

Only keeps the **last K** exchanges (sliding window):

```
┌─────────────────────────────────────────────────────────────────────┐
│             ConversationBufferWindowMemory (k=2)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conversation:                Memory keeps (ONLY LAST 2):           │
│  ┌───────────────────┐       ┌───────────────────────────────────┐ │
│  │ Turn 1 ─────────────────→ │ ╳ DROPPED (too old)               │ │
│  │ Turn 2 ─────────────────→ │ ╳ DROPPED (too old)               │ │
│  │ Turn 3 ─────────────────→ │ ╳ DROPPED (too old)               │ │
│  │ Turn 4 ─────────────────→ │ Human: "I live in Paris"          │ │
│  │ Turn 5 ─────────────────→ │ AI: "Paris is beautiful!"         │ │
│  └───────────────────┘       └───────────────────────────────────┘ │
│                                         ↑                          │
│                               Window slides forward                 │
│                                                                     │
│  ✅ PRO: Bounded size - never exceeds token limit                   │
│  ❌ CON: Forgets older context (user's name, preferences, etc.)     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=3)  # Keep last 3 exchanges
# After Turn 4, Turn 1 is forgotten
```

#### 4.3 ConversationSummaryMemory

Uses LLM to **summarize** the conversation (compressed memory):

```
┌─────────────────────────────────────────────────────────────────────┐
│                 ConversationSummaryMemory                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Full Conversation:              LLM Summarizes:                    │
│  ┌────────────────────────┐     ┌────────────────────────────────┐ │
│  │ "Hi, I'm Alice"        │     │                                │ │
│  │ "I work at Google"     │     │ "Alice is a software engineer  │ │
│  │ "I'm a software eng"   │ ──→ │  at Google working on AI       │ │
│  │ "I'm building AI"      │     │  projects. She's planning a    │ │
│  │ "Going to Japan soon"  │     │  trip to Japan and interested  │ │
│  │ "Love Tokyo temples"   │     │  in Tokyo temples."            │ │
│  └────────────────────────┘     └────────────────────────────────┘ │
│                                                                     │
│         ~200 tokens                    ~50 tokens                   │
│                                                                     │
│  ✅ PRO: Very compact - saves tokens, keeps essence                 │
│  ❌ CON: Loses exact details; costs extra LLM calls for summarizing │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
# LLM creates running summary - compact but loses detail
```

#### 4.4 ConversationSummaryBufferMemory

**Best of both worlds**: Recent messages + Summary of older ones:

```
┌─────────────────────────────────────────────────────────────────────┐
│              ConversationSummaryBufferMemory                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conversation History:                                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ OLD TURNS (summarized):                                       │  │
│  │ ┌────────────────────────────────────────────────────────┐   │  │
│  │ │ SUMMARY: "Alice is a Google engineer interested in AI   │   │  │
│  │ │          and planning a Japan trip."                    │   │  │
│  │ └────────────────────────────────────────────────────────┘   │  │
│  │                                                               │  │
│  │ RECENT TURNS (kept verbatim):                                 │  │
│  │ ┌────────────────────────────────────────────────────────┐   │  │
│  │ │ Human: "What temples should I visit?"                   │   │  │
│  │ │ AI: "Try Senso-ji in Tokyo and Fushimi Inari in Kyoto" │   │  │
│  │ │ Human: "How do I get there?"                            │   │  │
│  │ └────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ✅ PRO: Context preserved + bounded size                           │
│  ✅ PRO: Best for long conversations with important details         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=500  # Summarize when buffer exceeds this
)
# Summary of old + exact recent messages
```

#### 4.5 ConversationEntityMemory

Tracks **entities** (people, places, things) mentioned in conversation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 ConversationEntityMemory                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conversation:                    Entity Store:                     │
│  ┌────────────────────────┐      ┌─────────────────────────────┐   │
│  │"John works at Google"  │      │ JOHN:                       │   │
│  │"He's an ML engineer"   │  ──→ │   - Works at Google         │   │
│  │"Sarah is his colleague"│      │   - ML engineer             │   │
│  │"They build chatbots"   │      │   - Building chatbot        │   │
│  └────────────────────────┘      │                             │   │
│                                  │ SARAH:                      │   │
│                                  │   - Colleague of John       │   │
│                                  │   - Building chatbot        │   │
│                                  │                             │   │
│                                  │ GOOGLE:                     │   │
│                                  │   - Where John works        │   │
│                                  └─────────────────────────────┘   │
│                                                                     │
│  ✅ PRO: Perfect for tracking characters, projects, concepts        │
│  ❌ CON: Requires LLM calls to extract entities                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Memory Comparison Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MEMORY TYPE COMPARISON                           │
├───────────────────┬───────────────┬─────────────┬───────────────────┤
│ Memory Type       │ Token Usage   │ Recall      │ Best For          │
├───────────────────┼───────────────┼─────────────┼───────────────────┤
│ Buffer            │ ████████████  │ ★★★★★       │ Short chats       │
│                   │ (Unbounded)   │ (Perfect)   │                   │
├───────────────────┼───────────────┼─────────────┼───────────────────┤
│ BufferWindow      │ ████          │ ★★★☆☆       │ Long chats where  │
│                   │ (Fixed: k)    │ (Recent)    │ only recent       │
│                   │               │             │ matters           │
├───────────────────┼───────────────┼─────────────┼───────────────────┤
│ Summary           │ ██            │ ★★★☆☆       │ Very long chats   │
│                   │ (Compact)     │ (Gist)      │ with token limits │
├───────────────────┼───────────────┼─────────────┼───────────────────┤
│ SummaryBuffer     │ ██████        │ ★★★★☆       │ Long chats where  │
│                   │ (Balanced)    │ (Good)      │ both matter       │
├───────────────────┼───────────────┼─────────────┼───────────────────┤
│ Entity            │ ████          │ ★★★★☆       │ Tracking people,  │
│                   │ (Per entity)  │ (Entities)  │ places, concepts  │
└───────────────────┴───────────────┴─────────────┴───────────────────┘
```

### Memory Decision Flowchart

```
                    ┌─────────────────────┐
                    │ How long are your   │
                    │   conversations?    │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │   Short       │  │   Medium      │  │    Long       │
    │  (<10 turns)  │  │ (10-50 turns) │  │  (50+ turns)  │
    └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
            │                  │                  │
            ▼                  │                  │
    ┌───────────────┐          │                  │
    │ConversationBu-│          │                  │
    │  fferMemory   │          │                  │
    └───────────────┘          │                  │
                               ▼                  │
                    ┌─────────────────────┐       │
                    │ Need exact recent   │       │
                    │    messages?        │       │
                    └──────────┬──────────┘       │
                               │                  │
                    ┌──────────┼──────────┐       │
                    │          │          │       │
                    ▼          ▼          │       │
            ┌───────────┐ ┌─────────────┐ │       │
            │ Yes       │ │ No          │ │       │
            │           │ │             │ │       │
            └─────┬─────┘ └──────┬──────┘ │       │
                  │              │        │       │
                  ▼              ▼        │       │
          ┌──────────────┐ ┌──────────┐   │       │
          │SummaryBuffer │ │BufferWin-│   │       │
          │Memory        │ │dowMemory │   │       │
          └──────────────┘ └──────────┘   │       │
                                          │       │
                                          │       ▼
                                          │ ┌───────────┐
                                          │ │ Summary   │
                                          │ │ Memory    │
                                          │ └───────────┘
                                          │
                                          ▼
                              ┌─────────────────────┐
                              │Tracking specific    │
                              │entities? → Entity   │
                              │Memory               │
                              └─────────────────────┘
```

---

## 5. Chains

### What is a Chain?

A **chain** connects multiple components into a **pipeline**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WHAT IS A CHAIN?                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  A chain is like a factory assembly line:                           │
│                                                                     │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   │
│  │ Input  │ → │ Step 1 │ → │ Step 2 │ → │ Step 3 │ → │ Output │   │
│  └────────┘   └────────┘   └────────┘   └────────┘   └────────┘   │
│                                                                     │
│  In LangChain:                                                      │
│                                                                     │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   │
│  │Variables│→ │ Prompt │ → │  LLM   │ → │ Parser │ → │ Result │   │
│  │{"topic":│  │Template│   │        │   │        │   │        │   │
│  │ "AI"}   │  │        │   │        │   │        │   │        │   │
│  └────────┘   └────────┘   └────────┘   └────────┘   └────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### LCEL: LangChain Expression Language

The modern way to build chains using the **pipe operator** (`|`):

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LCEL - THE PIPE OPERATOR                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  prompt | llm | parser                                              │
│                                                                     │
│  Is equivalent to:                                                  │
│                                                                     │
│  def chain(input):                                                  │
│      step1 = prompt.format(input)                                   │
│      step2 = llm.invoke(step1)                                      │
│      step3 = parser.parse(step2)                                    │
│      return step3                                                   │
│                                                                     │
│  But LCEL is:                                                       │
│  ✅ More readable                                                   │
│  ✅ Handles async automatically                                     │
│  ✅ Supports streaming                                              │
│  ✅ Built-in error handling                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Types of Chains

#### 5.1 Simple Chain (Linear)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIMPLE CHAIN                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  {"topic": "AI"}                                                    │
│        │                                                            │
│        ▼                                                            │
│  ┌──────────────────────────────────────┐                          │
│  │ ChatPromptTemplate                    │                          │
│  │ "Tell me a fun fact about {topic}"   │                          │
│  └───────────────────┬──────────────────┘                          │
│                      │                                              │
│                      ▼                                              │
│  ┌──────────────────────────────────────┐                          │
│  │             LLM                       │                          │
│  │  AIMessage(content="Did you know...")│                          │
│  └───────────────────┬──────────────────┘                          │
│                      │                                              │
│                      ▼                                              │
│  ┌──────────────────────────────────────┐                          │
│  │        StrOutputParser               │                          │
│  │  "Did you know that AI can..."       │                          │
│  └──────────────────────────────────────┘                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Tell me about {topic}")
    | llm
    | StrOutputParser()
)

result = chain.invoke({"topic": "black holes"})
```

#### 5.2 Parallel Chain (RunnableParallel)

Run multiple operations **simultaneously**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                       PARALLEL CHAIN                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  {"topic": "Python"}                                                │
│         │                                                           │
│         ├─────────────────┬─────────────────┐                       │
│         │                 │                 │                       │
│         ▼                 ▼                 ▼                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │  Summary    │   │ Key Points  │   │ Use Cases   │               │
│  │  Prompt     │   │  Prompt     │   │  Prompt     │               │
│  │     ↓       │   │     ↓       │   │     ↓       │               │
│  │    LLM      │   │    LLM      │   │    LLM      │               │
│  │     ↓       │   │     ↓       │   │     ↓       │               │
│  │  Parser     │   │  Parser     │   │  Parser     │               │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘               │
│         │                 │                 │                       │
│         └─────────────────┼─────────────────┘                       │
│                           │                                         │
│                           ▼                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ {                                                              │ │
│  │   "summary": "Python is a versatile...",                       │ │
│  │   "key_points": "1. Easy syntax\n2. Large ecosystem...",       │ │
│  │   "use_cases": "Web development, Data science..."              │ │
│  │ }                                                              │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  All three run AT THE SAME TIME - faster!                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel(
    summary=prompt1 | llm | StrOutputParser(),
    key_points=prompt2 | llm | StrOutputParser(),
    use_cases=prompt3 | llm | StrOutputParser()
)

result = parallel_chain.invoke({"topic": "Python"})
# result["summary"], result["key_points"], result["use_cases"]
```

#### 5.3 Sequential Chain

Output of one step **feeds into** the next:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SEQUENTIAL CHAIN                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  {"topic": "Climate Change"}                                        │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Research                                             │   │
│  │ Prompt: "Research {topic} and provide 3 key insights"        │   │
│  │ Output: "1. Global temps rising... 2. Ice caps... 3. ..."    │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Outline (uses Step 1 output!)                        │   │
│  │ Prompt: "Based on {research}, create an outline"             │   │
│  │ Output: "I. Introduction\n II. Rising Temps\n III. ..."      │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: Write (uses Step 2 output!)                          │   │
│  │ Prompt: "Based on {outline}, write introduction"             │   │
│  │ Output: "Climate change is one of the most pressing..."      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Each step builds on the previous one!                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# Each step uses the output of the previous step
sequential_chain = (
    {"topic": RunnablePassthrough()}
    | RunnableParallel(
        topic=lambda x: x["topic"],
        research=research_prompt | llm | StrOutputParser()
    )
    | RunnableParallel(
        research=lambda x: x["research"],
        outline=outline_prompt | llm | StrOutputParser()
    )
    | RunnableParallel(
        outline=lambda x: x["outline"],
        final_article=write_prompt | llm | StrOutputParser()
    )
)
```

#### 5.4 Router Chain

Routes to **different chains** based on input:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ROUTER CHAIN                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User Question: "What is 15% of 240?"                               │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ CLASSIFIER: "What type of question is this?"                 │   │
│  │             → "math"                                         │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│         ┌────────────────────┼────────────────────┐                │
│         │                    │                    │                │
│    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐          │
│    │  math   │          │ science │          │ history │          │
│    │ route   │          │  route  │          │  route  │          │
│    └────┬────┘          └─────────┘          └─────────┘          │
│         │ ◄── Selected!                                            │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ MATH EXPERT PROMPT:                                          │   │
│  │ "You are a math tutor. Solve step by step: What is 15%..."   │   │
│  │                                                               │   │
│  │ OUTPUT: "Step 1: 15% = 0.15                                   │   │
│  │          Step 2: 0.15 × 240 = 36                              │   │
│  │          Answer: 36"                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Different questions → Different specialized handlers               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
def route_question(info):
    category = info["category"].lower()
    question = info["question"]

    if "math" in category:
        return math_prompt.format(question=question)
    elif "science" in category:
        return science_prompt.format(question=question)
    else:
        return general_prompt.format(question=question)

router_chain = (
    RunnableParallel(
        question=RunnablePassthrough(),
        category=classify_prompt | llm | StrOutputParser()
    )
    | RunnableLambda(route_question)
    | llm
    | StrOutputParser()
)
```

### Chain Comparison

| Chain Type | Pattern | Use Case |
|------------|---------|----------|
| **Simple** | A → B → C | Basic prompt → LLM → parse |
| **Parallel** | A → [B, C, D] → combine | Multiple analyses at once |
| **Sequential** | A → B (uses A's output) → C (uses B's output) | Multi-step reasoning |
| **Router** | A → (if X then B, if Y then C) | Different handling per category |

---

## 6. Putting It All Together

### Complete Example: Conversational Q&A Chain

```
┌─────────────────────────────────────────────────────────────────────┐
│              COMPLETE CONVERSATIONAL Q&A SYSTEM                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User: "What's the capital of France?"                              │
│                    │                                                │
│                    ▼                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 1. MEMORY                                                     │  │
│  │    Load chat history: [previous messages...]                  │  │
│  └───────────────────────────┬──────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 2. CHAT PROMPT TEMPLATE                                       │  │
│  │    ┌──────────────────────────────────────────────────────┐  │  │
│  │    │ System: You are a helpful assistant.                  │  │  │
│  │    │ History: {chat_history}                               │  │  │
│  │    │ Human: {question}                                     │  │  │
│  │    └──────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────┬──────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 3. LLM                                                        │  │
│  │    Generates response considering history                     │  │
│  └───────────────────────────┬──────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 4. OUTPUT PARSER                                              │  │
│  │    Extracts clean text from AIMessage                         │  │
│  └───────────────────────────┬──────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 5. MEMORY UPDATE                                              │  │
│  │    Save this exchange to memory                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  AI: "The capital of France is Paris."                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Quick Reference

### Prompt Templates Cheat Sheet

```python
# Basic template
PromptTemplate.from_template("Tell me about {topic}")

# Chat template with roles
ChatPromptTemplate.from_messages([
    ("system", "You are {role}"),
    ("human", "{question}")
])

# Few-shot template
FewShotPromptTemplate(
    examples=[{"input": "X", "output": "Y"}],
    example_prompt=example_template,
    prefix="Instructions",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)
```

### Output Parsers Cheat Sheet

```python
StrOutputParser()                    # → str
CommaSeparatedListOutputParser()     # → list
JsonOutputParser(pydantic_object=X)  # → dict
PydanticOutputParser(pydantic_object=X)  # → Pydantic object
```

### Memory Cheat Sheet

```python
ConversationBufferMemory()           # Full history
ConversationBufferWindowMemory(k=3)  # Last k exchanges
ConversationSummaryMemory(llm=llm)   # Summarized history
ConversationSummaryBufferMemory(     # Summary + recent
    llm=llm, max_token_limit=500)
```

### Chains Cheat Sheet (LCEL)

```python
# Simple chain
chain = prompt | llm | parser

# Parallel
chain = RunnableParallel(a=chain1, b=chain2)

# With custom function
chain = prompt | llm | RunnableLambda(my_function)

# Passthrough (keep original input)
chain = RunnableParallel(
    original=RunnablePassthrough(),
    processed=prompt | llm
)
```

---

## Key Takeaways

1. **Prompt Templates** = Reusable, structured prompts with variables
2. **Output Parsers** = Convert LLM text → structured data (lists, dicts, objects)
3. **Memory** = Give LLMs the ability to remember past conversations
4. **Chains** = Connect components into powerful pipelines

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE LANGCHAIN FORMULA                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│     PROMPT     +     LLM     +    PARSER    +    MEMORY             │
│       ↓              ↓            ↓              ↓                  │
│   Structure      Generate      Transform      Remember              │
│   the input      response      output         context               │
│                                                                     │
│                    All connected by CHAINS                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

*This guide accompanies the LangChain Complete Guide notebook. Practice the concepts hands-on to solidify your understanding!*
