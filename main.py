# main.py
from fastapi import FastAPI, HTTPException
from models.context import ModelStrategyContext
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
context = ModelStrategyContext()


@app.post("/predict")
async def predict(model_type: str, data: dict, params: dict = None):
    """统一预测接口"""
    start = time.time()

    try:
        result = await context.process(model_type, data, params)
        return {
            "success": True,
            "result": result,
            "model_type": model_type,
            "processing_time": time.time() - start
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(500, str(e))


@app.get("/models")
async def list_models():
    """查看所有模型"""
    return context.get_all_info()


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)