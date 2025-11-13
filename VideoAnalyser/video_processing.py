import cv2
import tempfile
import numpy as np
from VideoAnalyser.test_emotion import predict_emotion  # Import the real model

def process_video(video_bytes):
    # Save the incoming video bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)

    if not cap.isOpened():
        return {"error": "❌ Failed to open video file"}

    frame_count = 0
    emotions_detected = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ✅ Real emotion detection using the imported model
        try:
            emotion_label, confidence = predict_emotion(frame)
            emotions_detected.append({
                "emotion": emotion_label,
                "confidence": round(confidence, 2)
            })
        except Exception as e:
            emotions_detected.append({"error": str(e)})

        # Optional: Limit number of analyzed frames (for speed)
        if frame_count >= 5:
            break

    cap.release()

    return {
        "total_frames": frame_count,
        "frames_analyzed": len(emotions_detected),
        "emotion_analysis": emotions_detected
    }
