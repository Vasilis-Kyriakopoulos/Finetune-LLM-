import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Setup the App
app = FastAPI(title="Git Diff to Commit Message API")

# 2. Define Request Schema
# This ensures users send valid JSON with a "diff" field
class DiffRequest(BaseModel):
    diff: str
    max_tokens: int = 50  # Default value, but user can change it

# 3. Global Variables for Model (Loaded on Startup)
model = None
tokenizer = None
device = None

@app.on_event("startup")
def load_model():
    global model, tokenizer, device

    # Configuration
    MODEL_PATH = "data/best_model.pt" # Or your specific checkpoint path
    BASE_MODEL_NAME = "gpt2"     # Needed for the tokenizer configuration

    print("Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Tokenizer (Use the base gpt2 tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model Structure & Weights
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer)) # Important if you resized during training

    # Load your fine-tuned weights
    # map_location ensures it loads even if you move from GPU -> CPU
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}!")

# 4. Generation Endpoint
@app.post("/generate")
async def generate_commit_message(request: DiffRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 1. Tokenize the input diff
        inputs = tokenizer(request.diff, return_tensors="pt", truncation=True, max_length=512).to(device)

        # 2. Generate the output
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs, 
                max_new_tokens=request.max_tokens,
                pad_token_id=tokenizer.eos_token_id
            )

        # 3. Decode the result
        full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        # Simple split if you want to separate the input diff from the generated msg
        # This depends on how your prompt was structured during training
        answer = full_text.replace(request.diff, "").strip()

        return {
            "generated_message": answer,
            "full_text": full_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Health Check
@app.get("/health")
def health_check():
    return {"status": "ok", "device": str(device)}