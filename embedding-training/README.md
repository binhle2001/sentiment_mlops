# Embedding Training Service

This service is responsible for training a sentence embedding model based on labeled data from the database.

## Setup

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**

    Create a `.env` file in the `embedding-training` directory with the following content:

    ```
    DB_USER=your_db_user
    DB_PASSWORD=your_db_password
    DB_HOST=your_db_host
    DB_PORT=your_db_port
    DB_NAME=your_db_name
    ```

## Running the Service

To run the service, use the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

## Training the Model

To start the training process, send a POST request to the `/api/train` endpoint:

```bash
curl -X POST http://localhost:8001/api/train
```

The training process will run in the background. The trained model will be saved to the `./bge-m3-finetuned-transformer` directory, and the ONNX-exported model will be saved to the `./bge-m3-finetuned-transformer-onnx` directory.