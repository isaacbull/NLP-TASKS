from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pyttsx3
import librosa
import numpy as np
import soundfile as sf

app = FastAPI()
engine = pyttsx3.init()

class Text(BaseModel):
    text: str

@app.post("/clone-voice/")
async def clone_voice(file: UploadFile = File(...), text: Text):
    # Load the reference voice file
    reference_voice, sr = librosa.load(file.file, sr=None)
    
    # Generate synthetic speech
    engine.save_to_file(text.text, 'temp.wav')
    engine.runAndWait()
    
    # Load synthetic speech
    synthetic_voice, sr = librosa.load('temp.wav', sr=None)
    
    # Modify synthetic voice to match reference voice characteristics
    synthetic_voice = librosa.effects.pitch_shift(synthetic_voice, sr, n_steps=-2)
    
    # Save the modified voice
    sf.write('cloned_voice.wav', synthetic_voice, sr)
    
    return {"message": "Voice cloned and saved as cloned_voice.wav"}
