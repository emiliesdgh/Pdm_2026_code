import cv2
import mediapipe as mp
import numpy as np

# Define the solutions through the main 'mp' object
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# from ig_hand_state import get_fingers_state, get_hand_orientation, draw_cross_product_vector
import ig_hand_state as HS
import ig_temporal_gesture as temporal_gesture

TEXT_FLIPPED = True


def detect_hand_state():
    cap = cv2.VideoCapture(index=0)

    temporal_gesture_detection = temporal_gesture.TemporalGestureManager(window_size=15)

    # handStates = HS.HandState(None, None)  # Initialize with None, will be updated in the loop

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
                    if handStates.label == 'Left':
                        handStates.label = 'Right'
                    elif handStates.label == 'Right':
                        handStates.label = 'Left'


                    fingers_state = handStates.get_fingers_state(hand_landmarks)
                    hand_orientation = handStates.get_hand_orientation(hand_landmarks)

                    motion_detected, motion_type = temporal_gesture_detection.update(hand_landmarks, fingers_state, hand_orientation)

                    finger_state_rule1 = handStates.finger_flexion(hand_landmarks, finger_type='INDEX')
                    finger_state_rule3 = handStates.finger_contact(hand_landmarks, target_tip_idx=8)
                    
                    if fingers_state[0] == 1:
                        is_thumb_straight = True
                    finger_state_rule4 = handStates.thumb_direction(hand_landmarks, is_thumb_straight)

                    finger_state_rule5 = handStates.palm_orientation(frame, hand_landmarks, handStates.label)
                    finger_state_rule6 = handStates.hand_position(hand_landmarks)


                    if TEXT_FLIPPED:
                        frame = cv2.flip(frame, 1)

                    # Display the state and orientation on the image
                    # cv2.putText(frame, f'State: {fingers_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    # cv2.putText(frame, f'Orientation Angle: {hand_orientation}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    # cv2.putText(frame, f'Motion Detected: {motion_detected}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    # if motion_detected:
                    #     print(f"Motion Type: {motion_type}")
                    # else:
                    #     print("Stationary")

                    cv2.putText(frame, f'Finger Flexion Rule 1 (Index): {finger_state_rule1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f'Finger Contact Rule 3 (Thumb-Index) : {finger_state_rule3}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f'Thumb Direction Rule 4: {finger_state_rule4}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f'Palm Orientation Rule 5: {finger_state_rule5}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f'Hand Position Rule 6: {finger_state_rule6}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if TEXT_FLIPPED:
                        frame = cv2.flip(frame, 1)


                    # handStates.draw_cross_product_vector(frame, hand_landmarks)

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

                
            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cap.release()
                
if __name__ == "__main__":
    detect_hand_state()