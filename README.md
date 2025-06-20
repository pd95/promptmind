# PromptMind

PromptMind is a privacy-friendly, locally running AI knowledge base. It leverages Retrieval-Augmented Generation (RAG) techniques and integrates with [Ollama](https://ollama.com/), [FAISS](https://faiss.ai/), [LangChain](https://www.langchain.com/), and [LangGraph](https://github.com/langchain-ai/langgraph) to provide a secure, local environment for interacting with Large Language Models (LLMs) without sending your data to the cloud.  
All agent actions and pipeline steps can be traced and observed locally using OpenTelemetry and Jaeger – no data ever leaves your machine.

## Setup

### Prerequisites

- Python 3.10+
- Poetry
- [Ollama](https://ollama.com/) running locally with the following models:
  - [nomic-embed-text](https://ollama.com/library/nomic-embed-text)
  - [llama3.2](https://ollama.com/library/llama3.2)
  - [granite3.1-dense](https://ollama.com/library/granite3.1-dense)

     You can install each using the command `ollama pull <model>`

- (Optionally) Docker Desktop running locally (if you want to enable [Observability](#observability))

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

3. Ensure Ollama is running locally and [llama 3.2](https://ollama.com/library/llama3.2) is already downloaded (e.g. running `ollama pull llama3.2`).  
If you want to test/play with `scripts/basic_react_agent.py` you need "granite3.1-dense" (`ollama pull granite3.1-dense`), as other models didn't work well with this "old style ReAct agent script.

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

    - By default, the model `llama3.2` is used.
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

4. Enable tracing in `.env` file by setting

    ```sh
    TRACELOOP_TRACING_ENABLED=true
    ```

5. Running any script (in "scripts" folder) will now generate traces.

### How it works

- The OpenTelemetry Collector receives traces from your Python scripts (if instrumented).
- Jaeger provides a web UI to search, filter, and visualize traces, which helps you debug and optimize your AI pipelines.

### References

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)

---

## Project Structure

```tree
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

**Note:**  
PromptMind relies on several third-party open source components, including but not limited to [Ollama](https://github.com/ollama/ollama), [FAISS](https://github.com/facebookresearch/faiss), [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph), [OpenTelemetry](https://github.com/open-telemetry/opentelemetry-python), and [Jaeger](https://github.com/jaegertracing/jaeger).  
Each of these dependencies is distributed under its own license. Please consult their respective repositories for detailed license information and ensure compliance if you redistribute or modify this project.

---

**PromptMind**: Your private, local AI knowledge base.
