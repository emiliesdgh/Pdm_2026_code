# import cv2
print("--- START TEST ---")
import cv2
print("OpenCV loaded")
import mediapipe as mp
print("MediaPipe loaded")
import mp_HandGesture as HG
print("Local module loaded")
import numpy as np

# Define the solutions through the main 'mp' object
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

import mp_gesture as G

TEXT_FLIPPED = True

# this function is for testing the mp_HandGesture class: hand orientation, cross_product ==> hand position and orientation in frame
def run_hand_gesture_recognition():
    print("Starting hand tracking on webcam...")
    cap = cv2.VideoCapture(index=0)
    # print to show that the webcam is working
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    # handsClass = HG.HandGesture()

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame...")
                continue
                 
        # Do your hand logic here (e.g., compute vectors, draw cross product, etc.)
            # Check the frame for hands
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Draw the hand annotations on the image
            if results.multi_hand_landmarks:
                    
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    handsClass = HG.HandGesture(hand_landmarks, frame_rgb)
                    
                    handsClass.label = handedness.classification[0].label  # 'Left' or 'Right'
                    # label = handsClass.label
                    # Manually reverse the label IF USING WEBCAMERA
                    if handsClass.label == 'Left':
                        handsClass.label = 'Right'
                    elif handsClass.label == 'Right':
                        handsClass.label = 'Left'
                    ###############################################
                    print(f"Detected a {handsClass.label} hand")

                    ##### == hand gesture recognition with class == #####
                    gesture = handsClass.recognize_gesture(hand_landmarks)

                    cross, coordinates, palm_px, end_px = handsClass.cross_product(hand_landmarks)
                    thumb_PIXEL = coordinates[0]
                    index_PIXEL = coordinates[1]
                    middle_PIXEL =  coordinates[2]
                    ring_PIXEL =  coordinates[3]
                    pinky_PIXEL =  coordinates[4]
                    palm_PIXEL =  coordinates[5]


                    cv2.line(frame, tuple(palm_px[:2]), end_px, (0,0,255), 2) # Cross Product
                    # cv2.line(frame, tuple(palm_px[:2]), end_px2, (0,0,0), 2) # Cross Product
                    # cv2.line(frame, palm_pxRW, end_pxRW, (255,255,255), 3) # Cross Product RW
                    cv2.line(frame, tuple(palm_PIXEL[:2]), tuple(thumb_PIXEL[:2]), (0, 255, 0), 2)
                    cv2.line(frame, tuple(palm_PIXEL[:2]), tuple(middle_PIXEL[:2]), (255, 0, 0), 2)
                    # cv2.line(frame, tuple(palm_PIXEL[:2]), tuple(fingers_PIXEL[:2]), (255, 0, 0), 2)

                    ##### == hand position in frame == #####

                    alphaHorizontal, alphaVertical, X_real, Y_real, centerFrame = handsClass.displacement(hand_landmarks)
                    print("alphaHorizontal :", alphaHorizontal, "alphaVertical :", alphaVertical)
                    # print("X_real :", X_real, "Y_real :", Y_real)

                    handsClass.orientation(hand_landmarks)

                    # Draw the circle
                    cv2.circle(frame, centerFrame, radius=10, color=(255, 0, 255), thickness=-1)  # -1 = filled circle

     
                    if TEXT_FLIPPED:
                        frame = cv2.flip(frame, 1)

                    cv2.putText(
                        frame,
                        gesture,
                        (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4,
                        (255, 0, 0),
                        8,
                        )

                    if TEXT_FLIPPED:
                        frame = cv2.flip(frame, 1)

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


# this function is for testing the different hand gestures from the mp_gesture file ==> actual hand gestures (open palm, fist, peace sign, thumbs up, etc.)
def run_hand_tracking_on_webcam():
    print("Starting hand tracking on webcam...")
    cap = cv2.VideoCapture(index=0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame...")
                continue

            # Convert to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    
                    # Initialize your local class
                    handsClass = HG.HandGesture(hand_landmarks, frame_rgb)
                    
                    # Handle Handedness (Left vs Right)
                    handsClass.label = handedness.classification[0].label
                    if handsClass.label == 'Left':
                        handsClass.label = 'Right'
                    elif handsClass.label == 'Right':
                        handsClass.label = 'Left'
                    
                    print(f"Detected a {handsClass.label} hand")

                    # === INTEGRATED GESTURE RECOGNITION ===
                    # Use the advanced logic from mp_gesture.py
                    gesture = G.recognize_gesture2(hand_landmarks)

                    # Get coordinates for drawing
                    cross, coordinates, palm_px, end_px = handsClass.cross_product(hand_landmarks)
                    thumb_PIXEL = coordinates[0]
                    middle_PIXEL = coordinates[2]
                    palm_PIXEL = coordinates[5]

                    # Draw visual feedback
                    cv2.line(frame, tuple(palm_px[:2]), end_px, (0, 0, 255), 2) # Cross Product
                    cv2.line(frame, tuple(palm_PIXEL[:2]), tuple(thumb_PIXEL[:2]), (0, 255, 0), 2)
                    cv2.line(frame, tuple(palm_PIXEL[:2]), tuple(middle_PIXEL[:2]), (255, 0, 0), 2)

                    # Orientation and displacement logic
                    alphaHorizontal, alphaVertical, X_real, Y_real, centerFrame = handsClass.displacement(hand_landmarks)
                    handsClass.orientation(hand_landmarks)
                    cv2.circle(frame, centerFrame, radius=10, color=(255, 0, 255), thickness=-1)

                    # === DRAW GESTURE TEXT ===
                    # We flip temporarily to keep text readable if the whole frame is flipped later
                    if TEXT_FLIPPED:
                        frame = cv2.flip(frame, 1)

                    cv2.putText(
                        frame,
                        f"Gesture: {gesture}",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 0),
                        5,
                    )

                    if TEXT_FLIPPED:
                        frame = cv2.flip(frame, 1)

                    # Draw the standard MediaPipe landmarks
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Display the final frame
            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Running mp_main...")
    run_hand_tracking_on_webcam()