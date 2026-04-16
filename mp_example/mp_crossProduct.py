import mediapipe as mp
import cv2
import numpy as np

def cross_product(hand_landmarks, frame):

    landmarks = hand_landmarks.landmark
    height, width, depth = frame.shape

    # necessary landmarks
    palm = landmarks[0]
    thumb_tip = landmarks[4]
    middle_tip = landmarks[12]

    # Convert to pixel coordinates
    thumb_tipPIXEL = [int(thumb_tip.x * width), int(thumb_tip.y * height), int(thumb_tip.z * depth)]
    middle_tipPIXEL = [int(middle_tip.x * width), int(middle_tip.y * height), int(middle_tip.z * depth)]
    palm_PIXEL = [int(palm.x * width), int(palm.y * height), int(palm.z * depth)]
    coordinates = [thumb_tipPIXEL, middle_tipPIXEL, palm_PIXEL]

    # vectors for cross product
    v1 = np.array([thumb_tip.x - palm.x, thumb_tip.y - palm.y, thumb_tip.z - palm.z])
    v2 = np.array([middle_tip.x - palm.x, middle_tip.y - palm.y, middle_tip.z - palm.z])

    # cross product
    cross = np.cross(v1, v2)
    cross /= np.linalg.norm(cross)  # Normalize

    # Get palm base pixel coords
    palm_px = to_pixel_coords(palm, frame.shape)

    # Scale and project cross vector to draw
    scale = 100  # arbitrary visual scale
    end_px = (
        int(palm_px[0] + cross[0] * scale),
        int(palm_px[1] + cross[1] * scale)
    )

    return coordinates, palm_px, end_px


def to_pixel_coords(landmark, img_shape):
    h, w = img_shape[:2]
    return int(landmark.x * w), int(landmark.y * h)

