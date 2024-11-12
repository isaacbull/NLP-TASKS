from fastapi import FastAPI
from pydantic import BaseModel
from gtts import gTTS
import os

app = FastAPI()

class Text(BaseModel):
    text: str

@app.post("/synthesize/")
async def synthesize_text(text: Text):
    tts = gTTS(text.text, lang='en')
    tts.save("output.mp3")
    return {"message": "Text has been synthesized to speech and saved as output.mp3"}
