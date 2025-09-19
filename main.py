from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Chatbot.Chatbot import process_user_query
from Master_LLM.ML_Models.Inside_cave.genai import process_video_and_summarize

app = FastAPI(title="Gemini Mine Safety Bot API")

# ---- Enable CORS ----
origins = [
    "http://localhost:8080",  # React frontend
    "http://127.0.0.1:8080",
    "https://sih-nu-liart.vercel.app",
    # Add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, GET, OPTIONS etc.
    allow_headers=["*"],
)

# ---- Request/Response Models ----
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    final: str

# ---- Routes ----
@app.post("/chat", response_model=QueryResponse)
async def chat_with_bot(req: QueryRequest):
    result = process_user_query(req.query)
    return QueryResponse(**result)

# ---- Video Prediction Endpoint ----
@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    try:
        predictions = process_video_and_summarize(file.file)
        return predictions
    except Exception as e:
        return {
            "success": False,
            "error": f"❌ Video processing failed: {e}"
        }
    
# ---- Root Endpoint ----
@app.get("/")
async def root():
    return {"message": "✅ Gemini Mine Safety Bot API is running"}
