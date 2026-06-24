from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
from app.ml_models import analyze_sentiment, get_chatbot_response, recognize_image
from app.database import SessionLocal, RequestLog
import app.utils as utils

app = FastAPI(title="Corporate AI Platform")

# Пул потоков для тяжелых ИИ-задач, чтобы не блокировать сервер
executor = ThreadPoolExecutor(max_workers=4)

class TextRequest(BaseModel):
    text: str

def log_request(endpoint: str, input_data: str, output_data: str):
    db = SessionLocal()
    log = RequestLog(endpoint=endpoint, input_data=input_data, output_data=output_data)
    db.add(log)
    db.commit()
    db.close()

@app.post("/api/sentiment")
async def sentiment_analysis(req: TextRequest, bg_tasks: BackgroundTasks):
    # Предобработка текста
    clean_text = utils.preprocess_text(req.text)
    
    # Многопоточная обработка ИИ
    result = await app.state.loop.run_in_executor(executor, analyze_sentiment, clean_text)
    
    bg_tasks.add_task(log_request, "/api/sentiment", req.text, result)
    return {"sentiment": result}

@app.post("/api/chatbot")
async def chatbot_endpoint(req: TextRequest, bg_tasks: BackgroundTasks):
    result = await app.state.loop.run_in_executor(executor, get_chatbot_response, req.text)
    bg_tasks.add_task(log_request, "/api/chatbot", req.text, result)
    return {"response": result}

@app.post("/api/image-recognition")
async def image_endpoint(file: UploadFile = File(...), bg_tasks: BackgroundTasks = None):
    image = Image.open(io.BytesIO(await file.read()))
    result = await app.state.loop.run_in_executor(executor, recognize_image, image)
    bg_tasks.add_task(log_request, "/api/image", file.filename, str(result))
    return {"predictions": result}

@app.on_event("startup")
async def startup_event():
    import asyncio
    app.state.loop = asyncio.get_event_loop()