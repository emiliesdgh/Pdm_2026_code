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

                    motion_detected, spatial_motion, articulation = temporal_gesture_detection.update(hand_landmarks, finger_flexion_state, finger_contact_state, hand_orientation, hand_position)

                    prompt = get_symbolic_string_2(
                        global_vars, 
                        finger_flexion_state, finger_contact_state, hand_orientation, 
                        motion_detected, spatial_motion, articulation, hand_position, 
                        sensor_confidence, 
                        # environmental_context="ROBOT VISION: No object visible in the workspace. The floor is clear."   # test PICK_UP
                        environmental_context="ROBOT VISION: Large object in the path ahead. No clear path to navigate."    # test NAVIGATE_THERE
                        # environmental_context="ROBOT VISION: Object is visible and in reach. The path to navigate is clear."    # test STOP
                        # environmental_context="ROBOT VISION: Large object blocking the path ahead. No clear path to navigate."    # test SEARCH_AREA
                        # environmental_context="add simulated environmental context here"
                    )
                    # pass simulatedenvironmental context to test out the llm reasoning with the same gesture
                    # but different contexts (e.g., "The user is in a kitchen environment, standing near a table
                    # with various objects on it." vs "The user is in a living room environment, sitting on a couch
                    # with a coffee table in front of them.")

                    
                    ### === LLM Prompting with State Machine Logic === ###
                    if hri_state == "SLEEPING":
                        
                        if spatial_motion == "Stationary":
                            static_frame_count += 1
                        else:
                            static_frame_count = 0

                        if static_frame_count >= HOLD_TH:
                            hri_state = "LISTENING"
                            print("\n>>> SYSTEM AWAKE: Listening for command... <<<\n")
                            listening_timer = 45 # give user 1.5 seconds to make a gesture command
                            static_frame_count = 0  # Reset for next time

                            # -- Initialize a latching variable --
                            best_prompt = ""

                            # wipe the memory of the stationary wakeup hold
                            temporal_gesture_detection.gesture_history.clear()
                            temporal_gesture_detection.wrist_history.clear()

                            # reset the LLM state for the UI
                            llm_agent.current_intent = "Listening..."
                            llm_agent.current_confidence = 0.0
                            print("\n[SYSTEM AWAKE] - Listening for dynamic gesture command...\n")

                    elif hri_state == "LISTENING":
                        # RECORDING WINDOW
                        listening_timer -= 1

                        # -- Latching Logic ---
                        is_dynamic_articulation = "CLosing" in articulation or "Pinching" in articulation or "Grabbing" in articulation
                        is_dynamic_spatial = spatial_motion != "Stationary"

                        if is_dynamic_articulation or is_dynamic_spatial:
                            best_prompt = prompt  # Update the best prompt with the most recent dynamic gesture information


                        if listening_timer <= 0:
                            hri_state = "INFERENCING"
                            print("[SNAPSHOT TAKEN] - Sending to LLM")

                            # if they did a moving gesture, send the latched prompt
                            # if they just held a static pose, send the current prompt
                            final_prompt = best_prompt if best_prompt != "" else prompt

                            llm_agent.analyze_gesture_async(final_prompt)


                    elif hri_state == "INFERENCING":
                        if not llm_agent.is_inferencing:

                            # SAFETY GATE
                            if not llm_agent.is_inferencing:
                                intent = llm_agent.current_intent
                                confidence = getattr(llm_agent, 'current_confidence', 0.0)
                                target = getattr(llm_agent, 'current_target', 'None')
                                latency = getattr(llm_agent, 'current_latency', 0.0)    # to read the current latency

                                if confidence >= 0.75:
                                    print(f"[EXECUTE] -> {intent} on {target} (Confidence: {confidence}) | Latency: {latency:.2f}s")
                                    print(f"Protompt sent to LLM:\n{final_prompt}\n")
                                    # below line added but is necessary ?
                                    best_prompt = ""  # Clear the latched prompt after execution
                                else:
                                    print(f"[IGNORED] -> {intent} (Confidence too low: {confidence}) | Latency: {latency:.2f}s")

                                hri_state = "SLEEPING"
                                static_frame_count = 0  # Reset for next time


                    ### === Draw hand landmarks and info on the frame === ###

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

            if TEXT_FLIPPED:
                frame = cv2.flip(frame, 1)

            # --- DYNAMIC UI DISPLAY ---
            confidence = getattr(llm_agent, 'current_confidence', 0.0)
            intent = llm_agent.current_intent
            
            # If the LLM is currently thinking, show that
            if llm_agent.is_inferencing:
                display_text = "LLM: Thinking..."
                display_latency = ""
                text_color = (0, 255, 255) # Yellow
            # If it finished and passed the safety gate
            elif confidence >= 0.75 and intent != "Waiting...":
                display_text = f"EXECUTE: {intent} ({confidence:.2f}) "
                display_latency = f"Latency: {getattr(llm_agent, 'current_latency', 0.0):.2f}s"
                text_color = (0, 255, 0) # Green
            # If it finished but failed the safety gate
            elif confidence < 0.75 and intent != "Waiting...":
                display_text = f"IGNORED: {intent} (Conf: {confidence:.2f}) "
                display_latency = f"Latency: {getattr(llm_agent, 'current_latency', 0.0):.2f}s"
                text_color = (0, 0, 255) # Red
            # Default standby text
            else:
                display_text = f"LLM: {intent}"
                display_latency = ""
                text_color = (255, 255, 255) # White

            cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            cv2.putText(frame, display_latency, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            if TEXT_FLIPPED:
                frame = cv2.flip(frame, 1)

            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cap.release()
                
if __name__ == "__main__":

    detect_hand_state()


