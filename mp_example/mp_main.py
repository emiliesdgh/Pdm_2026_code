import cv2
import numpy as np

# import mediapipe.python.solutions.hands as mp_hands
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# import mediapipe.python.solutions.drawing_utils as mp_drawing
# import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

import mp_HandGesture as HG

TEXT_FLIPPED = True
### Add the fact that if the end effector rotates, then all the values have to be adapted

def run_hand_tracking_on_webcam():
    cap = cv2.VideoCapture(index=0)
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

                    # fingers_PIXEL = coordinates[6]
                    # # print("index_pixel :", index_PIXEL)
                    # # print("middle_pixel :", middle_PIXEL)
                    # # print("ring_pixel :", ring_PIXEL)
                    # # print("pinky_pixel :", pinky_PIXEL)
                    # print("fingers_pixel :", fingers_PIXEL)

                    # fingers_PIXEL = np.array([index_PIXEL[0] + middle_PIXEL[0] + ring_PIXEL[0] + pinky_PIXEL[0], index_PIXEL[1] + middle_PIXEL[1] + ring_PIXEL[1] + pinky_PIXEL[1],0])/4
                    # fingers_PIXEL = fingers_PIXEL[0], fingers_PIXEL[1]
                    # # print("PALPX :",palm_PIXEL[:2])
                    # print("finger_pixel :", fingers_PIXEL)


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

if __name__ == "__main__":
    run_hand_tracking_on_webcam()