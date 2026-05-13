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

                    motion_detected, motion_type = temporal_manager.update(hand_landmarks, finger_flexion, hand_orient, hand_pos)
                    
                    # --- MEMORY LOGIC ---
                    if motion_detected:
                        frames_since_motion = 0
                        if "Unknown" not in motion_type:
                            last_significant_motion = motion_type
                    else:
                        frames_since_motion += 1
                        # If you hold still for 1 second, it forgets the previous motion
                        if frames_since_motion > 30: 
                            last_significant_motion = "Stationary"

                    # Generate the string constantly in the background
                    prompt = get_symbolic_string_2(global_vars, finger_flexion, finger_contact, hand_orient, motion_detected, last_significant_motion, pos_text)

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

# import cv2
# import mediapipe as mp
# import numpy as np
# import json
# import time
# import os

# import ig_hand_state as HS
# import ig_temporal_gesture as temporal_gesture
# from ig_inference import get_symbolic_string_2
# from ig_global_variables import GlobalVariables

# # ==========================================
# # 1. SET THIS BEFORE YOU RUN THE SCRIPT
# # Change this based on the gesture you are testing
# # (e.g., "NAVIGATE_THERE", "PICK_UP", "STOP", "SEARCH_AREA")
# # ==========================================
# GROUND_TRUTH_INTENT = "NAVIGATE_THERE"  # Change this to the intent you want to record for

# DATASET_FILE = "gesture_dataset.json"

# def record_dataset():
#     # Load existing data so we don't overwrite previous recordings
#     gesture_dataset = []
#     if os.path.exists(DATASET_FILE):
#         with open(DATASET_FILE, "r") as f:
#             try:
#                 gesture_dataset = json.load(f)
#             except json.JSONDecodeError:
#                 gesture_dataset = []

#     print(f"Loaded {len(gesture_dataset)} existing records. Camera starting...")

#     cap = cv2.VideoCapture(index=0)
#     _, frame = cap.read()
#     global_vars = GlobalVariables(frame)
#     temporal_manager = temporal_gesture.TemporalGestureManager(global_vars, window_size=15)

#     mp_hands = mp.solutions.hands
#     mp_drawing = mp.solutions.drawing_utils

#     # # State Machine / Timer Variables
#     # static_frame_count = 0
#     # HOLD_THRESHOLD = 20  # Approx 1 second of holding still
#     # feedback_text = "Waiting for hand..."
#     # feedback_timer = 0

#     # State Machine / Timer Variables
#     static_frame_count = 0
#     HOLD_THRESHOLD = 20  
#     feedback_text = "Waiting for hand..."
#     # --- NEW VARIABLES ---
#     cooldown_timer = 0
#     COOLDOWN_MAX = 90  # 3 seconds at 30 FPS
#     last_significant_motion = "Stationary"

#     with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success: continue

#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(frame_rgb)

#             if results.multi_hand_landmarks:
#                 for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    
#                     # Hand tracking logic
#                     handStates = HS.HandState(global_vars, hand_landmarks)
#                     handStates.label = handedness.classification[0].label
#                     if handStates.label == 'Left': handStates.label = 'Right'
#                     elif handStates.label == 'Right': handStates.label = 'Left'

#                     finger_flexion = handStates.get_finger_flexion_state()
#                     hand_orient, _ = handStates.hand_orientation(frame, handStates.label)
#                     hand_pos, pos_text = handStates.hand_position() 
#                     finger_contact = handStates.get_finger_contact_state()

#                     motion_detected, motion_type = temporal_manager.update(hand_landmarks, finger_flexion, hand_orient, hand_pos)
                    
#                     # --- NEW: MEMORY LOGIC ---
#                     # If we are actively moving, save the name of the motion
#                     if motion_detected and "Unknown" not in motion_type:
#                         last_significant_motion = motion_type

#                     # If we have been sitting still for too long (e.g., 2 seconds after recording), clear the memory
#                     if static_frame_count > 40:
#                         last_significant_motion = "Stationary"

#                     # Get the string you want the LLM to see
#                     prompt = get_symbolic_string_2(global_vars, finger_flexion, finger_contact, hand_orient, motion_detected, last_significant_motion, pos_text)

#                     # # --- THE AUTO-RECORD LOGIC ---
#                     # if motion_detected:
#                     #     static_frame_count = 0
#                     #     feedback_text = "Moving..."
#                     # else:
#                     #     static_frame_count += 1
#                     #     feedback_text = f"Holding still... {static_frame_count}/{HOLD_THRESHOLD}"
                   
#                     # # --- THE AUTO-RECORD LOGIC ---
#                     # if motion_detected:
#                     #     static_frame_count = 0
#                     #     feedback_text = f"Moving: {last_significant_motion}" # Shows what the system is seeing
#                     # else:
#                     #     static_frame_count += 1
#                     #     feedback_text = f"Holding still... {static_frame_count}/{HOLD_THRESHOLD}"

#                     # # If hand is held still for 1 second, capture the data!
#                     # if static_frame_count == HOLD_THRESHOLD:
#                     #     data_point = {
#                     #         "id": int(time.time() * 1000),
#                     #         "ground_truth": GROUND_TRUTH_INTENT,
#                     #         "symbolic_string": prompt
#                     #     }
#                     #     gesture_dataset.append(data_point)
#                     #     print(f"RECORDED: {GROUND_TRUTH_INTENT} | Total Dataset Size: {len(gesture_dataset)}")
                        
#                     #     feedback_text = ">>> RECORDED! <<<"
#                     #     feedback_timer = 15 # Keep text on screen for a moment
#                     #     static_frame_count = 0 # Reset so it doesn't record 30 times a second
#                     # --- THE COOLDOWN & RECORD LOGIC ---
#                     if cooldown_timer > 0:
#                         # We just recorded! Ignore the user for 3 seconds so they can reset.
#                         cooldown_timer -= 1
#                         static_frame_count = 0
#                         feedback_text = f"RELAX & RESET HAND... ({cooldown_timer})"
#                     else:
#                         # Normal tracking
#                         if motion_detected:
#                             static_frame_count = 0
#                             feedback_text = f"Moving: {last_significant_motion}"
#                         else:
#                             static_frame_count += 1
#                             feedback_text = f"Holding still... {static_frame_count}/{HOLD_THRESHOLD}"

#                         # TRIGGER RECORDING
#                         if static_frame_count == HOLD_THRESHOLD:
#                             data_point = {
#                                 "id": int(time.time() * 1000),
#                                 "ground_truth": GROUND_TRUTH_INTENT,
#                                 "symbolic_string": prompt
#                             }
#                             gesture_dataset.append(data_point)
#                             print(f"RECORDED: {GROUND_TRUTH_INTENT} | Total Dataset Size: {len(gesture_dataset)}")
                            
#                             # Activate the 3-second cooldown!
#                             cooldown_timer = COOLDOWN_MAX
#                             static_frame_count = 0

#                     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # --- UI VISUALS ---
#             frame = cv2.flip(frame, 1)
            
#             # Display current target and feedback
#             cv2.putText(frame, f"Target Intent: {GROUND_TRUTH_INTENT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
#             # Change color if it just recorded
#             color = (0, 255, 0) if "RECORDED" in feedback_text else (0, 165, 255)
#             cv2.putText(frame, feedback_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
#             cv2.putText(frame, f"Total Data Points: {len(gesture_dataset)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#             cv2.putText(frame, "Press 'q' to Save & Quit", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#             # if feedback_timer > 0:
#             #     feedback_timer -= 1
#             #     if feedback_timer == 0:
#             #         feedback_text = "Waiting for movement..."

#             cv2.imshow("Dataset Recorder", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
    
#     cap.release()
#     cv2.destroyAllWindows()

#     # Save everything safely at the end
#     with open(DATASET_FILE, "w") as f:
#         json.dump(gesture_dataset, f, indent=4)
#     print(f"\n--- SUCCESS: Saved {len(gesture_dataset)} total records to {DATASET_FILE} ---\n")

# if __name__ == "__main__":
#     record_dataset()