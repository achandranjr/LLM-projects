# LLM-Projects

A collection of experiments, prototypes, and utilities involving Large Language Models (LLMs). This repository contains multiple sub-projects exploring prompt engineering, agent-based automation, fine-tuning, and retrieval-augmented generation (RAG), among other techniques. Each directory represents a standalone or modular experiment.



## 🧠 Highlighted Projects

### `computer_agent/`
An agent that uses screenshots of a computer environment to interpret tasks and interact with the browser. Implements tool dispatching, action recognition, and a modular architecture for web-based automation.

### `rag/`
A working RAG (Retrieval-Augmented Generation) system built from scratch. Features include:
- Chunked vector storage with similarity search
- Query handling and document retrieval
- Feedback and rating UI system

### `fine_tune/`
Contains scripts and datasets for fine-tuning LLMs on specialized instruction sets. Example: emulating historical figures' speech patterns or summarizing documents.

### `web_agent/`
Automates website interaction using Playwright and LLM tool selection. Supports secure logins using secrets managers like Bitwarden.

## 🛠 Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/achandranjr/LLM-projects.git
cd LLM-projects
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Project-specific setup**

Navigate into each subdirectory for individual instructions or README files (if provided). Some projects may require additional setup, such as API keys, browser drivers, or front-end builds.

## 🧪 Technologies Used

- **Python** (3.9+)
- **OpenAI / HuggingFace APIs**
- **Playwright**
- **Faiss / SentenceTransformers**
- **FastAPI / Flask**
- **Bitwarden SDK**
- **LLM prompt orchestration libraries**

## 🚧 Work in Progress

This repository is under active development. Not all features are fully documented or production-ready. Feel free to open issues or contribute if you'd like to collaborate.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
