import cv2


def list_camera_indexes():
    index = 0
    available_indexes = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        available_indexes.append(index)
        cap.release()
        index += 1
    return available_indexes


# List available camera indexes
camera_indexes = list_camera_indexes()
print("Available camera indexes:", camera_indexes)

# Connect to the external camera (e.g., index 1)
if len(camera_indexes) > 1:  # Assuming the external camera is at index 1
    external_camera_index = camera_indexes[1]
else:
    external_camera_index = (
        0  # Fallback to default webcam if no external camera is found
    )

cap = cv2.VideoCapture(external_camera_index)

if not cap.isOpened():
    print(f"Failed to open camera with index {external_camera_index}")
else:
    print(f"Connected to camera with index {external_camera_index}")
    cap.release()