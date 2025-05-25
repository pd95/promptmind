# PromptMind

PromptMind is a privacy-friendly, locally running AI knowledge base. It leverages Retrieval-Augmented Generation (RAG) techniques and integrates with [Ollama](https://ollama.com/), [FAISS](https://faiss.ai/), and [LangChain](https://www.langchain.com/) to provide a secure, local environment for interacting with Large Language Models (LLMs) without sending your data to the cloud.

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

2. Run the indexer:

    ```bash
    poetry run python scripts/index.py
    ```

    (by default the documents in `docs` will be read. But you can specify different folder, files and even URLs as a parameter.)

3. Run the query app:

    ```bash
    poetry run python scripts/query.py "What is this project about?"
    ```

    (by default the model "phi4-mini" will be used, but you can specify a different model as additional parameter)

4. The system will retrieve relevant information using FAISS and generate responses with your local LLM via Ollama.

## Project Structure

```
promptmind/
├── app.py
├── scripts/
│    └── index.py
├── docs/
├── vector_store/
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License.

---

**PromptMind**: Your private, local AI knowledge base.
