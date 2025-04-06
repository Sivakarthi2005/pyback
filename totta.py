import os
import cv2
import time
import json
import queue
import threading
import numpy as np
import pandas as pd
import pygame
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
from fuzzywuzzy import fuzz, process
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Input

# === Setup ===
audio_dir = "temp"
os.makedirs(audio_dir, exist_ok=True)

clients = set()
speech_queue = queue.Queue()
speech_lock = threading.Lock()
pygame.mixer.init()

# Load dataset
df = pd.read_csv("agriculture_questions_complete.csv")
df["question"] = df["question"].str.lower()

# Load emotion model
emotion_dict = {0: "Confused", 1: "Disgusted", 2: "Sleepy", 3: "Normal", 4: "Neutral", 5: "Disturbed", 6: "Thinking"}
model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
model.load_weights("model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
sleepy_count = 0
last_emotion = None

# FastAPI App
app = FastAPI()

# === Models ===
class SpeechInput(BaseModel):
    text: str

# === Helper Functions ===
def get_best_answer(question):
    best_match = process.extractOne(question.lower(), df["question"], scorer=fuzz.token_set_ratio)
    if best_match and best_match[1] >= 70:
        return df.loc[df["question"] == best_match[0], "answer"].values[0]
    return "I'm sorry, but I don't have an answer for that."

def speak(text):
    with speech_lock:
        from gtts import gTTS  # local import to prevent blocking startup
        tts = gTTS(text=text, lang='en')
        path = os.path.join(audio_dir, "output.mp3")
        tts.save(path)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

def emotion_detection():
    global last_emotion, sleepy_count
    cap = cv2.VideoCapture(0)
    last_emotion_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

            if time.time() - last_emotion_time > 10:
                prediction = model.predict(cropped_img, verbose=0)
                maxindex = int(np.argmax(prediction))
                last_emotion = emotion_dict[maxindex]
                last_emotion_time = time.time()

                if last_emotion == "Confused" or last_emotion == "Thinking":
                    speech_queue.put("Did you understand?")
                elif last_emotion == "Sleepy":
                    sleepy_count += 1
                    speech_queue.put("Please stay awake and focused.")
                    if sleepy_count >= 5:
                        speech_queue.put("You seem tired. Please take a break.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                elif last_emotion == "Disturbed":
                    speech_queue.put("Is something bothering you?")

            cv2.putText(frame, last_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def speech_worker():
    while True:
        msg = speech_queue.get()
        if msg is None:
            break
        speak(msg)
        speech_queue.task_done()

# === API Endpoints ===

@app.post("/speech-input")
async def speech_input(data: SpeechInput, background_tasks: BackgroundTasks):
    user_input = data.text.lower()

    if "bye" in user_input:
        response = "Goodbye! Have a fantastic day!"
        background_tasks.add_task(speak, response)
        return {"response": response}
    else:
        answer = get_best_answer(user_input)
        background_tasks.add_task(speak, answer)
        return {"response": answer}

@app.get("/emotion")
async def get_emotion():
    return {"emotion": last_emotion}

# === Startup Tasks ===

@app.on_event("startup")
def startup_event():
    threading.Thread(target=emotion_detection, daemon=True).start()
    threading.Thread(target=speech_worker, daemon=True).start()

@app.get("/")
async def root():
    return {"message": "API is running. Use /speech-input or /emotion."}
