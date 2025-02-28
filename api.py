import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from io import BytesIO
import numpy as np
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes
from fastapi.responses import StreamingResponse
import time

app = FastAPI(title="Zonos API", description="OpenAI-compatible TTS API for Zonos")

# Model Management
MODELS = {
    "transformer": None,
    "hybrid": None
}

VOICE_CACHE: Dict[str, torch.Tensor] = {}

def load_models():
    """Load both models at startup and keep them in VRAM"""
    device = "cuda"
    MODELS["transformer"] = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    MODELS["transformer"].requires_grad_(False).eval()
    MODELS["hybrid"] = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
    MODELS["hybrid"].requires_grad_(False).eval()

# API Models
class SpeechRequest(BaseModel):
    model: str = Field("Zyphra/Zonos-v0.1-transformer", description="Model to use")
    input: str = Field(..., max_length=500, description="Text to synthesize")
    voice: Optional[str] = Field(None, description="Voice ID to use")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speaking speed multiplier")
    language: str = Field("en-us", description="Language code")
    emotion: Optional[Dict[str, float]] = None
    response_format: str = Field("mp3", description="Audio format (mp3 or wav)")

class VoiceResponse(BaseModel):
    voice_id: str
    created: int  # Unix timestamp

# API Endpoints
@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    try:
        model = MODELS["transformer" if "transformer" in request.model else "hybrid"]

        # Convert speed to speaking_rate (15.0 is default)
        speaking_rate = 15.0 * request.speed

        # Prepare emotion tensor if provided
        emotion_tensor = None
        if request.emotion:
            emotion_values = [
                request.emotion.get("happiness", 1.0),
                request.emotion.get("sadness", 0.05),
                request.emotion.get("disgust", 0.05),
                request.emotion.get("fear", 0.05),
                request.emotion.get("surprise", 0.05),
                request.emotion.get("anger", 0.05),
                request.emotion.get("other", 0.1),
                request.emotion.get("neutral", 0.2)
            ]
            emotion_tensor = torch.tensor(emotion_values, device="cuda").unsqueeze(0)

        # Get voice embedding from cache if provided
        speaker_embedding = VOICE_CACHE.get(request.voice) if request.voice else None

        # Default conditioning parameters
        cond_dict = make_cond_dict(
            text=request.input,
            language=request.language,
            speaker=speaker_embedding,
            emotion=emotion_tensor,
            speaking_rate=speaking_rate,
            device="cuda",
            unconditional_keys=[] if request.emotion else ["emotion"]
        )

        conditioning = model.prepare_conditioning(cond_dict)

        # Generate audio
        codes = model.generate(
            prefix_conditioning=conditioning,
            max_new_tokens=86 * 30,
            cfg_scale=2.0,
            batch_size=1,
            sampling_params=dict(min_p=0.15)
        )

        wav_out = model.autoencoder.decode(codes).cpu().detach()
        sr_out = model.autoencoder.sampling_rate

        # Ensure proper shape
        if wav_out.dim() > 2:
            wav_out = wav_out.squeeze()
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)

        # Convert to requested format
        buffer = BytesIO()
        torchaudio.save(buffer, wav_out, sr_out, format=request.response_format)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type=f"audio/{request.response_format}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/voice")
async def create_voice(
    file: UploadFile = File(...),
    name: str = None
):
    try:
        # Read the audio file
        content = await file.read()
        audio_data = BytesIO(content)

        # Load and process audio
        wav, sr = torchaudio.load(audio_data)

        # Generate embedding using transformer model (handles GPU automatically)
        speaker_embedding = MODELS["transformer"].make_speaker_embedding(wav, sr)

        # Generate unique voice ID and cache embedding
        timestamp = int(time.time())
        voice_id = f"voice_{timestamp}_{len(VOICE_CACHE)}"
        VOICE_CACHE[voice_id] = speaker_embedding.to("cuda")  # Ensure it's on GPU

        return VoiceResponse(
            voice_id=voice_id,
            created=timestamp
        )

    except Exception as e:
        error_msg = str(e)
        if "cuda" in error_msg.lower():
            error_msg = "GPU error while processing voice. Please try again."
        elif "load" in error_msg.lower():
            error_msg = "Failed to load audio file. Please ensure it's a valid audio format."

        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@app.get("/v1/audio/models")
async def list_models():
    """List available models and their status"""
    return {
        "models": [
            {
                "id": "Zyphra/Zonos-v0.1-transformer",
                "created": 1234567890,
                "object": "model",
                "owned_by": "zyphra"
            },
            {
                "id": "Zyphra/Zonos-v0.1-hybrid",
                "created": 1234567890,
                "object": "model",
                "owned_by": "zyphra"
            }
        ]
    }

# Load models at startup
@app.on_event("startup")
async def startup_event():
    load_models()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4100)
