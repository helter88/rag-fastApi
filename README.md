# RAG Microservices with FastAPI and LangGraph

This project demonstrates a scalable, microservice-based Retrieval-Augmented Generation (RAG) system. It is built with Python, using FastAPI for the API layer, LangGraph for orchestrating the RAG pipeline, and ChromaDB as the vector store.

This project is designed as a portfolio piece to showcase skills in AI development, including building production-ready AI systems with modern tools and architectures.

## Project Overview

The system is composed of two main microservices:

1.  **Document Manager (`rag-document-manager-ms`)**: Handles the ingestion, processing, and storage of documents.
2.  **Query Service (`rag-query-ms`)**: Manages user queries, retrieves relevant information from the document store, and generates answers using a Large Language Model (LLM).

This separation of concerns allows for independent scaling and development of the ingestion and querying functionalities.

## Architecture

The overall architecture consists of three main components communicating via REST APIs:

```
+--------------------------+      +--------------------------+      +----------------------+
|   User / Client          |----->|   Query Service (FastAPI)|----->|   LLM (Google Gemini)|
+--------------------------+      |   (rag-query-ms)         |      +----------------------+
             ^                    |   - LangGraph Pipeline   |
             |                    |   - Retrieve & Generate  |
             |                    +------------+-------------+
             |                                 |
+--------------------------+                   | (Retrieval)
| Document Manager (FastAPI)|                  |
| (rag-document-manager-ms)|                   |
|  - Load & Parse Docs     |      +------------v-------------+
|  - Chunk & Embed         |----->|  Vector Store (ChromaDB) |
+--------------------------+      +--------------------------+

```

*   **`rag-document-manager-ms`**: This service exposes endpoints for uploading documents. It uses the `unstructured` and `PyPDF` libraries to load various document formats (PDF, DOCX, TXT), `langchain` to split them into manageable chunks, and a sentence-transformer model to create vector embeddings, which are then stored in ChromaDB.
*   **`rag-query-ms`**: This service receives user questions. It uses `LangGraph` to define a stateful RAG pipeline. The graph first retrieves relevant document chunks from ChromaDB based on the query, then passes the context and the original question to an LLM (Google Gemini) to generate a final answer. It also identifies the source documents for the generated answer.
*   **ChromaDB**: Acts as the central vector store, persisting document embeddings and allowing for efficient similarity searches. It is run as a separate container.

## Key Technologies

- **Backend Framework**: FastAPI
- **RAG Orchestration**: LangGraph
- **Core AI/ML Libraries**: LangChain, LangChain Community, LangChain Google GenAI
- **Vector Database**: ChromaDB
- **Document Processing**: Unstructured, PyPDF, LangChain Text Splitters
- **LLM Embeddings**: Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
- **LLM for Generation**: Google Gemini
- **Containerization**: Docker, Docker Compose
- **Async Server**: Uvicorn, Gunicorn

## Features

- **Microservice Architecture**: Scalable and maintainable design.
- **Multi-Format Document Ingestion**: Supports uploading and processing of PDF, DOCX, TXT, and EPUB files.
- **CRUD Operations for Documents**: Ingest, list, and delete documents from the vector store.
- **Stateful RAG Pipeline**: Orchestrated with LangGraph for robust and clear query processing logic.
- **Source Attribution**: The generated answer includes references to the source documents.
- **Asynchronous API**: Fully async FastAPI application for high performance.

## Setup & Running the Project

To run this project, you need to have Docker and Docker Compose installed.

1.  **Environment Variables**:
    You will need a Google API Key to use the Gemini model. Create a `.env` file in the `rag-query-ms` directory with the following content:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```

2.  **Build and Run the Services**:
    From the root directory of the project, run the following command:
    ```bash
    docker-compose up --build
    ```
    This command will build the Docker images for both microservices and start the ChromaDB container.

    - `rag-document-manager-ms` will be available at `http://localhost:8001`
    - `rag-query-ms` will be available at `http://localhost:8002`
    - `ChromaDB` will be available at `http://localhost:8000`

## API Endpoints

You can access the interactive API documentation (Swagger UI) for each service at:
- **Document Manager**: `http://localhost:8001/docs`
- **Query Service**: `http://localhost:8002/docs`

### Document Manager Service

- **`POST /ingest-to-rag`**: Upload one or more documents to be processed and stored in the RAG pipeline.
- **`GET /documents`**: List the filenames of all documents currently in the vector store.
- **`DELETE /documents/{filename}`**: Delete a document and all its associated data from the vector store.

### Query Service

- **`POST /query`**: Ask a question to the RAG system.

    **Request Body**:
    ```json
    {
      "question": "Your question here"
    }
    ```

    **Success Response**:
    ```json
    {
      "answer": "The generated answer from the LLM.",
      "sources": [
        "filename1.pdf",
        "filename2.docx"
      ]
    }
    ```
