import sys
import os

print(f"Python Path: {sys.executable}")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
    
    import mediapipe as mp
    # Test a specific component
    hands = mp.solutions.hands
    print("MediaPipe solutions loaded successfully")
    
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
except Exception as e:
    print(f"ANOTHER ERROR: {e}")