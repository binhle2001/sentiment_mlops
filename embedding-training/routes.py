from fastapi import APIRouter, BackgroundTasks
from starlette.responses import JSONResponse

from trainer import run_training

router = APIRouter()

@router.post("/train", status_code=202)
def train_model(background_tasks: BackgroundTasks):
    """
    Endpoint to trigger the model training process.
    """
    background_tasks.add_task(run_training)
    return JSONResponse(
        status_code=202,
        content={"message": "Model training started in the background."}
    )