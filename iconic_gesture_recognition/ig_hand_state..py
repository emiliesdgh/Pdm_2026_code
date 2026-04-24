import numpy as np
import cv2

### === Global for fingers === ###
# FINGER = (TIP, BASE)
THUMB = (4, 2)
INDEX = (8, 5)
MIDDLE = (12, 9)
RING = (16, 13)
PINKY = (20, 17)

WRIST = 0

### === Camera information === ###
## NEED to find the actial focal length of the camera for accurate real-world coordinate conversion
FOCAL_LENGTH = 1.0  # Focal length of the camera

def fingers_state(hand_landmarks):
    """
    Determine the hand state based on the fingers positions
    EXTENDED = 1 OR FOLDED = 0
    """

    landmarks = hand_landmarks.landmark

    def is_finger_extended(finger_tip, finger_base):
        return (
            landmarks[finger_tip].y < landmarks[finger_base].y
        )  # Higher y-value = lower in frame
    
    thumb = is_finger_extended(4, 2)
    index = is_finger_extended(8, 5)
    middle = is_finger_extended(12, 9)
    ring = is_finger_extended(16, 13)
    pinky = is_finger_extended(20, 17)

    fingers_state = [thumb, index, middle, ring, pinky]    

    return fingers_state

def hand_orientation(hand_landmarks):
# to test out because i'm not sure if it works as wished 
# if necessary --> check mp_HandGesture function palm_orientation
# not using any cross product perturbs me
    """
    Determine the hand orientation based on the wrist and middle finger base positions
    if facing UP, DOWN, LEFT, RIGHT and orientation
    """
    landmarks = hand_landmarks.landmark

    wrist = np.array([landmarks[WRIST].x, landmarks[WRIST].y])
    middle_base = np.array([landmarks[MIDDLE[1]].x, landmarks[MIDDLE[1]].y])

    # Vector from wrist to middle finger base
    vector = middle_base - wrist
    angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi

    return vector, angle


# to add : type of motion -- if STATIONNARY or MOVING
# to add : if MOVING --> discription of the motion (direction, speed, type of motion, etc.)