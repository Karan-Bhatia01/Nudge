import os

# --- CRITICAL: Force TensorFlow/Keras to use CPU only ---
# This must be done BEFORE importing tensorflow or keras
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Tells TensorFlow to not see any GPU devices
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses INFO and WARNING messages from TensorFlow C++ code

import cv2
import numpy as np
from tensorflow.keras.models import load_model # This import is now safe

# --- Fix model path for Linux deployment ---
# Get the directory of the current script (video_processing.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'model' directory inside VideoAnalyser
model_directory = os.path.join(current_script_dir, 'model')

# Construct the full path to the model file
# Use the _cpu_optimized.h5 if you re-saved it locally, otherwise keep original.
# Let's assume you've re-saved it for best compatibility:
model_file_path = os.path.join(model_directory, 'facialemotionmodel.h5') 
# If you haven't re-saved, use: model_file_path = os.path.join(model_directory, 'facialemotionmodel.h5')

# Load the trained model once
# Wrap in try-except for robust error logging during startup
try:
    model = load_model(model_file_path)
    print(f"✅ Model loaded successfully from: {model_file_path}")
except Exception as e:
    print(f"❌ Error loading model from {model_file_path}: {e}")
    # You might want to print contents of directory for debugging on Render:
    # print(f"Current working directory: {os.getcwd()}")
    # print(f"Contents of {current_script_dir}: {os.listdir(current_script_dir)}")
    # print(f"Contents of {model_directory}: {os.listdir(model_directory) if os.path.exists(model_directory) else 'Directory not found'}")
    raise # Re-raise the exception to crash early if model is crucial for app startup

# Define the emotion labels corresponding to the model's output classes
# Your labels: ['Angry', 'Disgust', 'Fear', 'Happy', 'Anxious', 'Surprise', 'Neutral', 'Confident']
# Standard FER models often use: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
# Ensure your EMOTIONS list exactly matches the order of your model's 8 output classes.
# If your model was trained on 7 standard emotions + Neutral, then "Anxious" and "Confident" are custom.
# Make sure the order matches your training.
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Anxious', 'Surprise', 'Neutral', 'Confident'] # Verify this order with your model's actual output classes

def preprocess_frame(frame, target_size=(48, 48)):
    """
    Preprocess a video frame:
    - Convert to grayscale
    - Resize to target size
    - Normalize pixel values
    - Reshape for model input
    """
    # Ensure frame is not None or empty
    if frame is None or frame.size == 0:
        return None # Or raise an error, depending on desired behavior

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, target_size[0], target_size[1], 1))
    return reshaped

def predict_emotion(frame):
    """
    Predict emotion for a given frame.

    Returns:
        emotion_label (str): The predicted emotion name
        confidence (float): The confidence score between 0 and 1
    """
    processed = preprocess_frame(frame)
    if processed is None:
        # Return default or error for empty frames
        return "No Face/Frame", 0.0

    # Ensure verbose=0 to prevent verbose TF output during prediction
    # This also helps avoid excessive logging on Render
    predictions = model.predict(processed, verbose=0)[0] 
    
    top_index = np.argmax(predictions)
    emotion_label = EMOTIONS[top_index]
    confidence = float(predictions[top_index])
    return emotion_label, confidence