import os
import threading
from fastapi import FastAPI
from config import get_settings
from database import init_pool, close_pool
from routes import router as api_router
from trainer import run_training

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

def train_if_model_does_not_exist():
    """Checks if the model exists and runs training if it doesn't."""
    # Check for a key file that indicates training is complete
    model_file = os.path.join(settings.output_dir, "pytorch_model.bin")
    if not os.path.exists(model_file):
        print("Model not found. Starting training process in the background...")
        try:
            run_training()
            print("Training process completed successfully.")
        except Exception as e:
            print(f"An error occurred during training: {e}")
    else:
        print(f"Model found at {settings.output_dir}. Skipping training.")

@app.on_event("startup")
async def startup_event():
    init_pool()
    # Run the training check in a background thread to avoid blocking the server
    training_thread = threading.Thread(target=train_if_model_does_not_exist)
    training_thread.start()

@app.on_event("shutdown")
async def shutdown_event():
    close_pool()

app.include_router(api_router, prefix="/api")