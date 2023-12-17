from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import dlib
from imutils import face_utils
import os

app = FastAPI()
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return "Sleeping"
    elif 0.21 < ratio <= 0.25:
        return "Drowsy"
    else:
        return "Active"

def process_image(image_path):
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        return {"user_status": "No user"}

    user_status = "Active"
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == "Sleeping" or right_blink == "Sleeping":
            user_status = "Sleeping"
        elif left_blink == "Drowsy" or right_blink == "Drowsy":
            user_status = "Drowsy"

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    return {"user_status": user_status}

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    # Save the uploaded file
    image_path = f"temp/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(file.file.read())

    # Process the image and get the user status
    result = process_image(image_path)

    # Return the result
    return JSONResponse(content=result)

