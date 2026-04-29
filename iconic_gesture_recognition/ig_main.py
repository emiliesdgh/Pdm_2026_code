import cv2
import mediapipe as mp
import numpy as np

# Define the solutions through the main 'mp' object
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

import ig_hand_state as HS
import ig_temporal_gesture as temporal_gesture
from ig_inference import get_symbolic_string

TEXT_FLIPPED = True

FINGERS ={
    "name": ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"],
    "tip_idx": [4, 8, 12, 16, 20],
    "dip_idx": [3, 7, 11, 15, 19],
    "pip_idx": [2, 6, 10, 14, 18],
    "base_idx": [1, 5, 9, 13, 17]
}

def detect_hand_state():
    cap = cv2.VideoCapture(index=0)
    temporal_gesture_detection = temporal_gesture.TemporalGestureManager(window_size=15)

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Check the frame for hands
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Draw the hand annotations on the image
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    handStates = HS.HandState(hand_landmarks.landmark, frame)  # Update hand state with current landmarks and frame
                    handStates.label = handedness.classification[0].label  # 'Left' or 'Right'
                    # label = handsClass.label
                    # Manually reverse the label IF USING WEBCAMERA
                    # might need to remove with Robot camera !!!
                    if handStates.label == 'Left':
                        handStates.label = 'Right'
                    elif handStates.label == 'Right':
                        handStates.label = 'Left'

                    
                    ### === Get the hand state information for sending to the LLM for symbolic representation === ###
                    finger_flexion_state = handStates.get_finger_flexion_state(hand_landmarks)
                    hand_orientation, hand_orientation_angle = handStates.hand_orientation(frame, hand_landmarks, handStates.label)
                    hand_position, pos_to_center = handStates.hand_position(hand_landmarks)
                    finger_contact_state = handStates.get_finger_contact_state(hand_landmarks)


                    motion_detected, motion_type = temporal_gesture_detection.update(hand_landmarks, finger_flexion_state, hand_orientation, hand_position)

                
                    is_thumb_straight = (finger_flexion_state[0] == 1)
                    finger_state_rule4 = handStates.thumb_direction(hand_landmarks, is_thumb_straight)


                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    get_symbolic_string(finger_flexion_state, finger_contact_state, hand_orientation, motion_detected, motion_type, hand_position)

                
            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cap.release()
                
if __name__ == "__main__":
    detect_hand_state()