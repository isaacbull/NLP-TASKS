from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from transformers import Tacotron2Tokenizer, Tacotron2ForConditionalGeneration, Wav2Vec2Processor
import librosa
import numpy as np
import soundfile as sf

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Tacotron2 model and tokenizer
tokenizer = Tacotron2Tokenizer.from_pretrained("NVIDIA/tacotron2")
model = Tacotron2ForConditionalGeneration.from_pretrained("NVIDIA/tacotron2").to(device)

class Text(BaseModel):
    text: str

@app.post("/clone-voice/")
async def clone_voice(file: UploadFile = File(...), text: Text):
    # Load the reference voice file
    reference_voice, sr = librosa.load(file.file, sr=None)
    
    # Tokenize text
    inputs = tokenizer(text.text, return_tensors="pt").to(device)
    
    # Generate synthetic speech
    with torch.no_grad():
        speech = model.generate(**inputs)

    # Convert tensor to numpy array
    synthetic_voice = speech.cpu().numpy().squeeze()
    
    # Modify synthetic voice to match reference voice characteristics
    synthetic_voice = librosa.effects.pitch_shift(synthetic_voice, sr, n_steps=-2)
    
    # Save the modified voice
    sf.write('cloned_voice.wav', synthetic_voice, sr)
    
    return {"message": "Voice cloned and saved as cloned_voice.wav"}
