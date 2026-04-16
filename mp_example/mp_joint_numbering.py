import cv2
import numpy as np

# import mediapipe.python.solutions.hands as mp_hands
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# import mediapipe.python.solutions.drawing_utils as mp_drawing
# import mediapipe.python.solutions.drawing_styles as mp_drawing_styles


TEXT_FLIPPED = True


def run_hand_tracking_on_webcam():
    cap = cv2.VideoCapture(index=0)

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
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # Loop through landmarks and draw numbers
                    for idx, landmark in enumerate(hand_landmarks.landmark):

                        if TEXT_FLIPPED:
                            h, w, _ = frame.shape  # Get frame dimensions
                            x, y = int(landmark.x * w), int(
                                landmark.y * h
                            )  # Convert to pixel coordinates

                            # Create a blank image to draw mirrored text
                            text = str(idx)
                            (text_w, text_h), _ = cv2.getTextSize(
                                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                            )

                            # Create blank text image
                            text_img = np.zeros(
                                (text_h + 5, text_w + 5, 3), dtype=np.uint8
                            )

                            # Draw text onto the blank image
                            cv2.putText(
                                text_img,
                                text,
                                (0, text_h),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )

                            # Flip the text image horizontally
                            text_img = cv2.flip(text_img, 1)

                            # Overlay the flipped text onto the original frame
                            # Ensure the text does not go outside the frame
                            if 0 <= x < w - text_w and 0 <= y - text_h < h:
                                roi = frame[
                                    y - text_h : y, x : x + text_w
                                ]  # Region of interest
                                text_img = cv2.resize(
                                    text_img, (roi.shape[1], roi.shape[0])
                                )  # Resize text if needed
                                frame[y - text_h : y, x : x + text_w] = cv2.addWeighted(
                                    roi, 1, text_img, 1, 0
                                )
                        else:
                            h, w, _ = frame.shape  # Get frame dimensions
                            x, y = int(landmark.x * w), int(
                                landmark.y * h
                            )  # Convert to pixel coordinates
                            cv2.putText(
                                frame,
                                str(idx),
                                (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                5,
                                (0, 255, 0),
                                10,
                                cv2.LINE_AA,
                            )

            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()


if __name__ == "__main__":
    run_hand_tracking_on_webcam()