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

# from mp_gesture import recognize_gesture, recognize_gesture2, get_finger_states, cross_product_vector, get_index_pointing_vector
from mp_gesture import recognize_gesture2, get_finger_states, cross_product_vector, get_index_pointing_vector
import mp_temporal_gesture as MTG

TEXT_FLIPPED = True
### === Camera information === ###
## NEED to find the actial focal length of the camera for accurate real-world coordinate conversion
FOCAL_LENGTH = 1.0  # Focal length of the camera

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
                    # print(f"Detected a {handsClass.label} hand")

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
                    # print("alphaHorizontal :", alphaHorizontal, "alphaVertical :", alphaVertical)
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
    cap = cv2.VideoCapture(index=0)

    # added the temporal gesture manager for pinch detection
    temporal_gesture_manager = MTG.TemporalGestureManager(window_size=15)

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

            # Check the frame for hands
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Draw the hand annotations on the image
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    static_gesture = recognize_gesture2(hand_landmarks, frame_rgb)
                    final_gesture = temporal_gesture_manager.update(hand_landmarks, static_gesture)

                    if final_gesture == "Pointing":
                        # 1. Calculate the 3D pointing direction
                        pointer_vec = get_index_pointing_vector(hand_landmarks)

                        # 2. Get the Index Tip position in pixels
                        # Use the frame dimensions from frame_rgb (height=H, width=W)
                        H, W, _ = frame_rgb.shape
                        tip_px = (int(hand_landmarks.landmark[8].x * W), 
                                int(hand_landmarks.landmark[8].y * H))

                        # 3. Project the vector onto the 2D frame
                        # Multiply the vector by a large number (e.g., 1000) to act as the laser length
                        laser_end_px = (
                            int(tip_px[0] + pointer_vec[0] * 1000),
                            int(tip_px[1] + pointer_vec[1] * 1000)
                        )

                        # 4. Drawing: Only show the laser if the gesture is "Pointing" 
                        # (You can refine your recognize_gesture2 to return "Pointing" for state [0, 1, 0, 0, 0])
                        cv2.line(frame, tip_px, laser_end_px, (0, 255, 0), 2)  # Neon green line
                        cv2.circle(frame, tip_px, 8, (0, 0, 255), -1)         # Red dot at the fingertip

                        _, cross_vect, coordinates = cross_product_vector(hand_landmarks, frame_rgb)
                        # print("cross_vect", cross_vect)
                        palm_basePIXEL = coordinates[4]
                        thumb_tipPIXEL = coordinates[1]
                        finger_tipPIXEL = coordinates[3]
                        scale = 0.1  # Adjust this based on your needs
                        end_point = [int(palm_basePIXEL[0] + scale * FOCAL_LENGTH),int(palm_basePIXEL[1] - scale * FOCAL_LENGTH)]
                        # print("main cross_vect", cross_vect)  
                        cv2.line(frame, tuple(palm_basePIXEL[:2]), tuple(thumb_tipPIXEL[:2]), (0, 255, 0), 2)
                        cv2.line(frame, tuple(palm_basePIXEL[:2]), tuple(finger_tipPIXEL[:2]), (255, 0, 0), 2)
                        cv2.line(frame, tuple(palm_basePIXEL[:2]), tuple(cross_vect), (0, 0, 255), 2)

                    if TEXT_FLIPPED:
                        frame = cv2.flip(frame, 1)

                    if final_gesture != "Pinch Performed" and final_gesture != "Stop Action":

                        cv2.putText(
                            frame,
                            final_gesture,
                            (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2,
                        )
                    else:
                        cv2.putText(
                            frame,
                            final_gesture,
                            (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            4,
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


if __name__ == "__main__":
    print("Running mp_main...")
    # run_hand_tracking_on_webcam()
    run_hand_gesture_recognition()