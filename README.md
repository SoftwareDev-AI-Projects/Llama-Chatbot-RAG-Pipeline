# Llama Chat Bot on Amazon Reviews RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline using FastAPI to create a chat bot that answers questions based on Amazon reviews.

## Features

- **FastAPI**: A modern, fast (high-performance), web framework for building APIs with Python 3.6+.
- **RAG Pipeline**: Combines retrieval and generation to provide accurate and contextually relevant answers.
- **Environment Variables**: Uses `dotenv` to manage environment variables.
- **Logging**: Integrated logging for better traceability and debugging.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/rajtulluri/Llama-chat-bot-on-Amazon-reviews-RAG-pipeline.git
    cd Llama-chat-bot-on-Amazon-reviews-RAG-pipeline
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the server**:
    ```sh
    uvicorn main:app --host 0.0.0.0 --port 8080
    ```

2. **Send a POST request to `/query` endpoint**:
    - **Endpoint**: `/query`
    - **Method**: POST
    - **Request Body**:
        ```json
        {
            "question": "Your question here",
            "product_name": "Product name here"
        }
        ```

## Project Structure

- `main.py`: The main entry point of the application.
- `rag/rag_pipeline.py`: Contains the implementation of the RAG pipeline.
- `templates/request.py`: Defines the request model.
- `templates/response.py`: Defines the response model.
