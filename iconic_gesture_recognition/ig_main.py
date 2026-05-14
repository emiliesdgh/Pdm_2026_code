import cv2
import mediapipe as mp
import numpy as np

# Define the solutions through the main 'mp' object
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

import ig_hand_state as HS
import ig_temporal_gesture as temporal_gesture
from ig_inference import get_symbolic_string_2

from ig_global_variables import GlobalVariables
from ig_llm_agent import LLMInferenceAgent

TEXT_FLIPPED = True

# def detect_hand_state(global_vars):
def detect_hand_state():
    cap = cv2.VideoCapture(index=0)
    _, frame = cap.read()
    global_vars = GlobalVariables(frame)
    temporal_gesture_detection = temporal_gesture.TemporalGestureManager(global_vars, window_size=15)

    # Initialize the LLM agent (make sure to have run 'ollama run mixtral' in the terminal first)
    llm_agent = LLMInferenceAgent()

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        # State Machine Variables
        hri_state = "SLEEPING"  # States: SLEEPING, LISTENING, INFERENCING
        static_frame_count = 0
        HOLD_TH = 30  # Number of frames to consider a gesture as "held" (tune based on frame rate)

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

                    # Extract sensor confidence (0.0 - 1.0) & pass it to the string generator
                    sensor_confidence = handedness.classification[0].score

                    handStates = HS.HandState(global_vars, hand_landmarks)  # Update hand state with current landmarks and frame
                    handStates.label = handedness.classification[0].label  # 'Left' or 'Right'

                    # Manually reverse the label IF USING WEBCAMERA
                    # might need to remove with Robot camera !!!
                    if handStates.label == 'Left':
                        handStates.label = 'Right'
                    elif handStates.label == 'Right':
                        handStates.label = 'Left'
                    
                    ### === Get the hand state information for sending to the LLM for symbolic representation === ###
                    finger_flexion_state = handStates.get_finger_flexion_state()
                    hand_orientation, hand_orientation_angle = handStates.hand_orientation(frame, handStates.label)
                    hand_position, pos_to_center = handStates.hand_position()
                    finger_contact_state = handStates.get_finger_contact_state()

                    motion_detected, motion_type = temporal_gesture_detection.update(hand_landmarks, finger_flexion_state, hand_orientation, hand_position)

                    prompt = get_symbolic_string_2(
                        global_vars, 
                        finger_flexion_state, finger_contact_state, hand_orientation, 
                        motion_detected, motion_type, hand_position, 
                        sensor_confidence
                    )

                    print(f"Status: {motion_type} | Detected: {motion_detected} | Inferencing: {llm_agent.is_inferencing} ")

                    ### === LLM Prompting with State Machine Logic === ###

                    # 1. Define the Wake Gesture (e.g., Open Palm facing camera)
                    # If all fingers extended (1) and facing Inward
                    is_wake_gesture = (sum(finger_flexion_state) == 5 and hand_orientation == 'Inward')

                    # 2. State Machine Logic
                    if hri_state == "SLEEPING":
                        if is_wake_gesture:
                            hri_state = "LISTENING"
                            print("\n>>> SYSTEM AWAKE: Listening for command... <<<\n")
                            
                    elif hri_state == "LISTENING":
                        if motion_detected:
                            static_frame_count = 0  # Reset timer if moving
                        else:
                            static_frame_count += 1 # Count stable frames

                        # Trigger LLM if user holds a stable pose for 1 second
                        if static_frame_count >= HOLD_TH and not llm_agent.is_inferencing:
                            print("\n>>> GESTURE LOCKED: Sending to LLM... <<<\n")
                            hri_state = "INFERENCING"
                            llm_agent.analyze_gesture_async(prompt)
                            static_frame_count = 0
                            
                    elif hri_state == "INFERENCING":
                        if not llm_agent.is_inferencing:
                            # --- THE LIVE SAFETY GATE ---
                            # Extract the LLM's final reasoning
                            intent = llm_agent.current_intent
                            confidence = getattr(llm_agent, 'current_confidence', 0.0)
                            target = getattr(llm_agent, 'current_target', 'None')

                            # Only execute if the LLM is highly confident
                            if confidence >= 0.75:
                                print(f"\n[ROBOT COMMAND EXECUTE] -> {intent} on {target} (Confidence: {confidence})")
                                # This is where you will eventually call your robot's movement API
                                # e.g., robot_controller.send_command(intent, target)
                            else:
                                print(f"\n[ROBOT COMMAND IGNORED] -> {intent} (Confidence too low: {confidence})")
                                # The robot stays safe and does nothing
                            # LLM is done, go back to sleep
                            hri_state = "SLEEPING"

                    ###===

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

            if TEXT_FLIPPED:
                frame = cv2.flip(frame, 1)

            # # Display the LLM's prediction directly on the screen
            # cv2.putText(frame, f"LLM: {llm_agent.current_prediction}", (10, 40), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- DYNAMIC UI DISPLAY ---
            confidence = getattr(llm_agent, 'current_confidence', 0.0)
            intent = llm_agent.current_intent
            
            # If the LLM is currently thinking, show that
            if llm_agent.is_inferencing:
                display_text = "LLM: Thinking..."
                text_color = (0, 255, 255) # Yellow
            # If it finished and passed the safety gate
            elif confidence >= 0.75 and intent != "Waiting...":
                display_text = f"EXECUTE: {intent} ({confidence:.2f})"
                text_color = (0, 255, 0) # Green
            # If it finished but failed the safety gate
            elif confidence < 0.75 and intent != "Waiting...":
                display_text = f"IGNORED: {intent} (Conf: {confidence:.2f})"
                text_color = (0, 0, 255) # Red
            # Default standby text
            else:
                display_text = f"LLM: {intent}"
                text_color = (255, 255, 255) # White

            cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
            if TEXT_FLIPPED:
                frame = cv2.flip(frame, 1)

            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cap.release()
                
if __name__ == "__main__":

    detect_hand_state()