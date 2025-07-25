from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import time
from .inference_server import InferenceServer
from .model_registry import ModelRegistry

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    context: Optional[str] = ""
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7

class QuestionResponse(BaseModel):
    answer: str
    processing_time: float

class ModelInfo(BaseModel):
    model_name: str
    version: str
    status: str

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token verification (replace with proper authentication)."""
    token = credentials.credentials
    # In production, verify the token against your authentication system
    if token != "your-secret-token":  # Replace with proper token validation
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

# Initialize FastAPI app
app = FastAPI(
    title="AI Engineer Pipeline API",
    description="API for fine-tuned language model inference",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables (in production, use proper dependency injection)
inference_server: Optional[InferenceServer] = None
model_registry = ModelRegistry()

@app.on_event("startup")
async def startup_event():
    """Initialize the inference server on startup."""
    global inference_server
    logging.basicConfig(level=logging.INFO)
    
    # In production, load model configuration from environment variables
    # For now, we'll use a placeholder
    # inference_server = InferenceServer("meta-llama/Llama-2-7b-hf", "./results/final_checkpoint")
    logging.info("API server started")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "AI Engineer Pipeline API is running"}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "inference_server_loaded": inference_server is not None,
        "timestamp": time.time()
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    token: str = Depends(verify_token)
):
    """Answer a question using the fine-tuned model."""
    if inference_server is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference server not initialized"
        )
    
    start_time = time.time()
    
    try:
        answer = inference_server.answer_question(
            question=request.question,
            context=request.context
        )
        
        processing_time = time.time() - start_time
        
        return QuestionResponse(
            answer=answer,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/models", response_model=List[ModelInfo])
async def list_models(token: str = Depends(verify_token)):
    """List all registered models."""
    try:
        models = model_registry.list_models()
        return [
            ModelInfo(
                model_name=model["model_name"],
                version=model["version"],
                status=model["status"]
            )
            for model in models
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )

@app.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    version: str = "latest",
    token: str = Depends(verify_token)
):
    """Load a specific model version for inference."""
    global inference_server
    
    try:
        model_info = model_registry.get_model(model_name, version)
        
        # In production, implement proper model loading logic
        # inference_server = InferenceServer(base_model_name, model_info["model_path"])
        
        return {
            "message": f"Model {model_name}:{version} loaded successfully",
            "model_path": model_info["model_path"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

