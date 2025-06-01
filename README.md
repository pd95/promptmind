# PromptMind

PromptMind is a privacy-friendly, locally running AI knowledge base. It leverages Retrieval-Augmented Generation (RAG) techniques and integrates with [Ollama](https://ollama.com/), [FAISS](https://faiss.ai/), [LangChain](https://www.langchain.com/), and [LangGraph](https://github.com/langchain-ai/langgraph) to provide a secure, local environment for interacting with Large Language Models (LLMs) without sending your data to the cloud.

## Setup

### Prerequisites

- Python 3.10+
- Poetry
- [Ollama](https://ollama.com/) installed and running locally

### Installation

1. Clone this repository:

    ```sh
    git clone https://github.com/pd95/promptmind.git
    cd promptmind
    ```

2. Install dependencies:

    ```sh
    poetry install
    ```

3. Ensure Ollama is running locally and [Phi-4-mini](https://ollama.com/library/phi4-mini) is already downloaded (e.g. running `ollama pull phi4-mini`)

### Usage

1. Drop your documents into the `docs` folder.

2. **Index your documents** (create or update the vector store):

    ```bash
    poetry run python scripts/index.py [<folder>|<file>|<url> ...] [--chunk_size=600] [--overlap=100]
    ```

    - By default, all documents in `docs` are indexed.
    - You can specify one or more folders, files, or URLs as sources.
    - Optional parameters: `--chunk_size` and `--overlap` control text splitting.

3. **Query your knowledge base**:

    ```bash
    poetry run python scripts/query.py "<your question>" [model]
    ```

    - By default, the model `phi4-mini` is used.
    - You can specify a different model as a second argument.

4. **Agents** (interactive chatbots):

    - **Basic Chatbot** (no web search):

        ```bash
        poetry run python agents/basic_chatbot.py [--model=<model>]
        ```

        - Default model: `phi4`
        - Simple interactive chat with a local LLM.

    - **Chatbot with Web Search**:

        ```bash
        poetry run python agents/chatbot_with_websearch.py [--model=<model>]
        ```

        - Default model: `granite3.1-dense:8b`
        - Adds DuckDuckGo web search as a tool for the agent.

---

## Scripts Overview

- **scripts/index.py**  
  Indexes documents from folders, files, or URLs into a FAISS vector store. Supports PDF, Markdown, and text files. Allows chunk size and overlap configuration via CLI.

- **scripts/query.py**  
  Simple RAG pipeline: retrieves relevant document chunks from the vector store and generates an answer using a local LLM via Ollama.

---

## Agents Overview

> **Note:** The agents use [LangGraph](https://github.com/langchain-ai/langgraph) for advanced conversational and tool-calling capabilities.

- **agents/basic_chatbot.py**  
  Minimal interactive chatbot using a local Ollama model. No tools or web search.

- **agents/chatbot_with_websearch.py**  
  Interactive chatbot with DuckDuckGo web search tool. Uses LangGraph and supports tool-calling for up-to-date answers.

---

## Observability

While developing AI scripts and agents, it is useful to see "what is going on" inside your LLM pipelines.  
PromptMind includes an [OpenTelemetry](https://opentelemetry.io/) and [Jaeger](https://www.jaegertracing.io/) setup for distributed tracing and observability.

The `opentelemetry` folder contains a Docker Compose setup that will spin up:
- **OpenTelemetry Collector** (for collecting and exporting traces)
- **Jaeger UI** (for visualizing traces and spans)

### How to start observability stack

1. Make sure you have [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed.
2. In the project root, run:

    ```sh
    cd opentelemetry
    docker compose up -d
    ```

3. Open the Jaeger UI in your browser: [http://localhost:16686](http://localhost:16686)

### How it works

- The OpenTelemetry Collector receives traces from your Python scripts (if instrumented).
- Jaeger provides a web UI to search, filter, and visualize traces, which helps you debug and optimize your AI pipelines.

### References

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)

---

## Project Structure

```
promptmind/
├── app.py
├── scripts/
│   ├── index.py
│   └── query.py
├── agents/
│   ├── basic_chatbot.py
│   └── chatbot_with_websearch.py
├── docs/
├── vector_store/
├── opentelemetry/
│   ├── docker-compose.yml
│   └── otel-collector-config.yml
├── pyproject.toml
└── README.md
```

## License

This project is licensed under the MIT License.

---

**PromptMind**: Your private, local AI knowledge base.
