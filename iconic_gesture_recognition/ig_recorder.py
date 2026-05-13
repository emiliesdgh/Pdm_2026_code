import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os

import ig_hand_state as HS
import ig_temporal_gesture as temporal_gesture
from ig_inference import get_symbolic_string_2
from ig_global_variables import GlobalVariables

# ==========================================
# 1. SET THIS BEFORE YOU RUN THE SCRIPT
# (e.g., "NAVIGATE_THERE", "PICK_UP", "STOP", "SEARCH_AREA")
# ==========================================
GROUND_TRUTH_INTENT = "SEARCH_AREA"  # Change this to the intent you want to record for

DATASET_FILE = "gesture_dataset_good.json"

def record_dataset():
    gesture_dataset = []
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r") as f:
            try:
                gesture_dataset = json.load(f)
            except json.JSONDecodeError:
                gesture_dataset = []

    print(f"Loaded {len(gesture_dataset)} existing records. Camera starting...")

    cap = cv2.VideoCapture(index=0)
    _, frame = cap.read()
    global_vars = GlobalVariables(frame)
    temporal_manager = temporal_gesture.TemporalGestureManager(global_vars, window_size=15)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Memory variables for manual recording
    last_significant_motion = "Stationary"
    frames_since_motion = 0
    feedback_text = "Press SPACEBAR to record"
    feedback_timer = 0

    with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            prompt = ""  # Default empty prompt

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    
                    handStates = HS.HandState(global_vars, hand_landmarks)
                    handStates.label = handedness.classification[0].label
                    if handStates.label == 'Left': handStates.label = 'Right'
                    elif handStates.label == 'Right': handStates.label = 'Left'

                    finger_flexion = handStates.get_finger_flexion_state()
                    hand_orient, _ = handStates.hand_orientation(frame, handStates.label)
                    hand_pos, pos_text = handStates.hand_position() 
                    finger_contact = handStates.get_finger_contact_state()

                    motion_detected, spatial_motion, articulation = temporal_manager.update(hand_landmarks, finger_flexion, hand_orient, hand_pos)
                    
                    # --- MEMORY LOGIC ---
                    if motion_detected:
                        frames_since_motion = 0
                        if "Unknown" not in spatial_motion:
                            last_significant_motion = spatial_motion
                    else:
                        frames_since_motion += 1
                        # If you hold still for 1 second, it forgets the previous motion
                        if frames_since_motion > 30: 
                            last_significant_motion = "Stationary"

                    # Generate the string constantly in the background
                    prompt = get_symbolic_string_2(global_vars, finger_flexion, finger_contact, hand_orient, motion_detected, last_significant_motion, articulation, pos_text)
                    # prompt = get_symbolic_string_2(global_vars, finger_flexion, finger_contact, hand_orient, motion_detected, spatial_motion, articulation, pos_text)

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- UI VISUALS ---
            frame = cv2.flip(frame, 1)
            
            cv2.putText(frame, f"Target Intent: {GROUND_TRUTH_INTENT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            color = (0, 255, 0) if "RECORDED" in feedback_text else (0, 165, 255)
            cv2.putText(frame, feedback_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show you what the computer is currently thinking the motion is
            cv2.putText(frame, f"Motion memory: {last_significant_motion}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            cv2.putText(frame, f"Total Data Points: {len(gesture_dataset)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "[SPACEBAR] Record | [Q] Save & Quit", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if feedback_timer > 0:
                feedback_timer -= 1
                if feedback_timer == 0:
                    feedback_text = "Press SPACEBAR to record"

            cv2.imshow("Dataset Recorder (Manual Mode)", frame)
            
            # ==========================================
            # KEYBOARD CONTROLS
            # ==========================================
            key = cv2.waitKey(1) & 0xFF
            
            # If Spacebar is pressed AND a hand is on screen
            if key == 32 and prompt != "": 
                data_point = {
                    "id": int(time.time() * 1000),
                    "ground_truth": GROUND_TRUTH_INTENT,
                    "symbolic_string": prompt
                }
                gesture_dataset.append(data_point)
                print(f"RECORDED: {GROUND_TRUTH_INTENT} | Total Dataset Size: {len(gesture_dataset)}")
                
                feedback_text = ">>> RECORDED! <<<"
                feedback_timer = 15 # Flash text for half a second
                
            elif key == ord("q"):
                break
    
    cap.release()
    cv2.destroyAllWindows()

    with open(DATASET_FILE, "w") as f:
        json.dump(gesture_dataset, f, indent=4)
    print(f"\n--- SUCCESS: Saved {len(gesture_dataset)} total records to {DATASET_FILE} ---\n")

if __name__ == "__main__":
    record_dataset()
