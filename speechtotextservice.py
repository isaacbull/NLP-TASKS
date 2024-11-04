from fastapi import FastAPI, UploadFile, File
import speech_recognition as sr
from pydub import AudioSegment

app = FastAPI()
recognizer = sr.Recognizer()

@app.post("/speech-to-text/")
async def speech_to_text(file: UploadFile = File(...)):
    audio = AudioSegment.from_file(file.file)
    audio.export("temp.wav", format="wav")
    
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        
    return {"text": text}
