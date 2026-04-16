import numpy as np
import cv2

### === Global for fingers === ###
# FINGER = (TIP, BASE)
THUMB = (4, 2)
INDEX = (8, 5)
MIDDLE = (12, 9)
RING = (16, 13)
PINKY = (20, 17)

### === Camera information === ###
FOCAL_LENGTH = 1.0  # Focal length of the camera


# TRIAL METHOD
def recognize_gesture(hand_landmarks):
    """
    Simple gesture detection based on landmark positions.
    """
    landmarks = hand_landmarks.landmark

    def is_finger_extended(finger_tip, finger_base):
        return (
            landmarks[finger_tip].y < landmarks[finger_base].y
        )  # Higher y-value = lower in frame

    thumb_extended = is_finger_extended(4, 2)
    index_extended = is_finger_extended(8, 5)
    middle_extended = is_finger_extended(12, 9)
    ring_extended = is_finger_extended(16, 13)
    pinky_extended = is_finger_extended(20, 17)

    if not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Fist"
    elif (
        index_extended and middle_extended and not ring_extended and not pinky_extended
    ):
        return "Peace Sign"
    elif all([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Full Palm"
    return "Unknown"

def all_fingers_below_palm(finger_tip, finger_base):
    finger_tip.y
    THUMB[0].y # y coordinate of tip of thumb

def project_point(point3D, cx, cy):
    x, y, z = point3D
    if z == 0:
        z = 1e-6  # prevent divide by zero
    u = int((x * FOCAL_LENGTH / abs(z)) + cx)
    v = int((y * FOCAL_LENGTH / abs(z)) + cy)
    return (u, v)


# FINGER STATE ANALYSIS
# Function to check if fingers are extended
def get_finger_states(hand_landmarks):
    """
    Returns a list indicating if each finger is extended (1) or folded (0).
    Uses relative landmark positions instead of just y-coordinates.
    """
    finger_states = []

    # Define landmarks for fingertips and PIP joints
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    pip_joints = [6, 10, 14, 18]  # Corresponding PIP joints

    for tip, pip in zip(finger_tips, pip_joints):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            finger_states.append(1)  # Finger is extended
        else:
            finger_states.append(0)  # Finger is folded

    # Thumb: Check distance instead of y-coordinates
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]

    if abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_ip.x - thumb_mcp.x):
        finger_states.insert(0, 1)  # Thumb extended
    else:
        finger_states.insert(0, 0)  # Thumb folded

    return finger_states


# Function to recognize gestures based on finger states
def recognize_gesture2(hand_landmarks, frame):
    finger_states = get_finger_states(hand_landmarks)
    print("Finger states: ", finger_states)
    organic_gesture, cross_vec, coordinates = cross_product_vector(hand_landmarks, frame)
    # """
    # Identifies gestures based on which fingers are extended or folded.
    # """
    if organic_gesture:
        ### NOT VERY ROBUST
        if organic_gesture:#cross_vec:
            return "Palm Up"
        elif not organic_gesture:#cross_vec:
            return "Palm Down"
    elif not organic_gesture:
        if finger_states == [1, 1, 1, 1, 1]:
            return "Open Palm"
        if finger_states == [0, 0, 0, 0, 0]:
            return "Fist"
        elif finger_states == [1, 1, 1, 1, 1]:
            return "Open Palm"
        elif finger_states == [0, 1, 1, 0, 0]:
            return "Peace Sign"
        elif finger_states == [1, 0, 0, 0, 0]:
            return "Thumbs Up"
        elif finger_states == [1, 1, 0, 0, 1]:
            return "Rock Sign"
        elif finger_states == [1, 1, 1, 0, 0]:
            return "Three Fingers"
        else:
            return "Unknown Gesture"