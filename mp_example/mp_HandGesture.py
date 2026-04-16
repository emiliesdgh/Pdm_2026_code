# import mediapipe as mp
import cv2
import numpy as np



### === Class === ###
class HandGesture:
    def __init__(self, hand_landmarks, frame):
        self.landmarks = hand_landmarks
        self.label = None

        ### === Camera information === ###
        self.frame = frame
        self.H = frame.shape[0] # Height
        self.W = frame.shape[1] # Width
        self.D = frame.shape[2] # Depth

        self.FOV = 1.32  # radians
        self.FOCAL_LENGTH = self.W / (2 * np.tan(self.FOV / 2))  # Focal length of the camera
        self.centerFrame = (self.W // 2, self.H // 2)

        self.camera_view = np.array([0, 0, -1])

    def get_label(self):
        return self.label
    
    def to_pixel_coords(self, hand_landmarks):
        self.landmarks = hand_landmarks#.landmark
        return int(self.landmarks.x * self.W), int(self.landmarks.y * self.H), int(self.landmarks.z * self.D)
    
    def displacement(self, hand_landmarks):
        self.landmarks = hand_landmarks.landmark
        """
        Function to calculate the displacement in angle of the hand in the center of the frame
        """
        centerHand = self.landmarks[0]
        # Convert to pixel coordinates
        centerHand_px = self.to_pixel_coords(centerHand)

        # Amount of pixels to move
        dx = self.centerFrame[0] - centerHand_px[0]
        dy = centerHand_px[1] - self.centerFrame[1]
        # print("dx: ", dx, "dy: ", dy)

        X_realCam = dx * self.D / self.FOCAL_LENGTH # LEFT RIGHT --> if - then left of center point, if + then right of center point
        Y_realCam = dy * self.D / self.FOCAL_LENGTH # UP DOWN --> if - then above center point, if + then below center point
        # print("X_real: ", X_realCam, "Y_real: ", Y_realCam)

        alphaHorizontal = np.arctan2(dx, self.FOCAL_LENGTH)  # Angle in radians
        alphaHorizontal = np.degrees(alphaHorizontal)

        alphaVertical = np.arctan2(dy, self.FOCAL_LENGTH)
        alphaVertical = np.degrees(alphaVertical)

        ## X in robot frame stays the same equal to Z in camera,
        ## Y is on the side equal to X in camera
        Y_real = X_realCam # correct direction ? or needs to be - X_realCam
        ## Z is up and down equal to Y in camera
        Z_real = Y_realCam # correct direction ? or needs to be - Y_realCam
        
        return alphaHorizontal, alphaVertical, Y_real, Z_real, self.centerFrame

    def orientation(self, hand_landmarks):
        self.landmarks = hand_landmarks.landmark
        """
        Function to calculate the orientation of the hand
        """
        cross, _, _, _ = self.cross_product(hand_landmarks)

        dot = np.dot(cross, self.camera_view)
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)
        ### Not as robust as needed maybe, the orientation seems a bit off
        
        # print("Angle in degrees: ", angle_deg)

        return angle_deg

    def cross_product(self, hand_landmarks):
        self.landmarks = hand_landmarks.landmark

        palm = self.landmarks[0]
        thumb_tip = self.landmarks[4]

        index_tip = self.landmarks[8]
        middle_tip = self.landmarks[12]
        ring_tip = self.landmarks[16]
        pinky_tip = self.landmarks[20]

        # fingers_tip = np.sum(index_tip, middle_tip, ring_tip, pinky_tip)
        # Convert to pixel coordinates
        thumb_tipPIXEL = self.to_pixel_coords(thumb_tip)

        index_tipPIXEL = self.to_pixel_coords(index_tip)
        middle_tipPIXEL = self.to_pixel_coords(middle_tip)
        ring_tipPIXEL = self.to_pixel_coords(ring_tip)
        pinky_tipPIXEL = self.to_pixel_coords(pinky_tip)
        # print("index_PIXEL :", index_tipPIXEL)

        # fingers_tipPIXEL = self.to_pixel_coords(fingers_tip)
        fingers_PIXEL = (index_tipPIXEL + middle_tipPIXEL + ring_tipPIXEL + pinky_tipPIXEL)
        # print("fingers_PIXEL :", fingers_PIXEL)

        palm_PIXEL = self.to_pixel_coords(palm)
        coordinates = [thumb_tipPIXEL, index_tipPIXEL, middle_tipPIXEL, ring_tipPIXEL, pinky_tipPIXEL, palm_PIXEL]#, fingers_PIXEL]

        # vectors for cross product
        vthumb = np.array([thumb_tip.x - palm.x, thumb_tip.y - palm.y, thumb_tip.z - palm.z])
        vindex = np.array([index_tip.x - palm.x, index_tip.y - palm.y, index_tip.z - palm.z])
        vmiddle = np.array([middle_tip.x - palm.x, middle_tip.y - palm.y, middle_tip.z - palm.z])
        vring = np.array([ring_tip.x - palm.x, ring_tip.y - palm.y, ring_tip.z - palm.z])
        vpinky = np.array([pinky_tip.x - palm.x, pinky_tip.y - palm.y, pinky_tip.z - palm.z])

        # self.label = HandGesture.get_label(self)
        vfingers = (vindex+vmiddle+vring+vpinky)/4
        cross = np.cross(vthumb, vfingers)
        # print("CROSS : ", cross)

        if self.label == 'Right':
            print("Right hand")
            # print("CROSS : ", cross)
            cross = cross#np.cross(v1, v2)
            # cross2 = np.cross(vthumb, vmiddle)
        elif self.label == 'Left':
            print("Left hand")
            # print("CROSS : ", cross)
            cross = -cross#np.cross(v2, v1)
            # cross2 = np.cross(vmiddle,vthumb)

        cross /= np.linalg.norm(cross)
        # cross2 /= np.linalg.norm(cross2)

        # Get palm base pixel coords
        # palm_px = to_pixel_coords(palm)

        # Scale and project cross vector to draw
        scale = 500

        end_px = (int(palm_PIXEL[0] + cross[0] * scale),
                  int(palm_PIXEL[1] + cross[1] * scale))
        
        # end_px2 = (int(palm_PIXEL[0] + cross2[0] *scale),
                #   int(palm_PIXEL[1] + cross2[1] *scale))
        
        return cross, coordinates, palm_PIXEL, end_px#, end_px2

    def palm_orientation(self, hand_landmarks):
        cross, _, _, _ = self.cross_product(hand_landmarks)
        crossX, crossY, crossZ = cross

        angle = np.arctan2(crossY, crossX) * 180 / np.pi

        angle_rad = np.arccos(np.clip(np.dot(cross, self.camera_view), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        if angle > 25 and angle < 155:
            return True, False # Palm DOWN
        elif angle < -25 and angle > -155:
            return True, True # Palm UP
        else:
            return False, False
        
    def get_finger_states(self, hand_landmarks):
        """
        Returns a list indicating if each finger is extended (1) or folded (0).
        Uses relative landmark positions instead of just y-coordinates.
        """
        finger_states = []
        self.landmarks = hand_landmarks.landmark

        # Define landmarks for fingertips and PIP joints
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        pip_joints = [6, 10, 14, 18]  # Corresponding PIP joints

        for tip, pip in zip(finger_tips, pip_joints):
            if self.landmarks[tip].y < self.landmarks[pip].y:
                finger_states.append(1)  # Finger is extended
            else:
                finger_states.append(0)  # Finger is folded

        # Thumb: Check distance instead of y-coordinates
        thumb_tip = self.landmarks[4]
        thumb_ip = self.landmarks[3]
        thumb_mcp = self.landmarks[2]

        if abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_ip.x - thumb_mcp.x):
            finger_states.insert(0, 1)  # Thumb extended
        else:
            finger_states.insert(0, 0)  # Thumb folded

        return finger_states

    # Function to recognize gestures based on finger states
    def recognize_gesture(self, hand_landmarks):
        finger_states = self.get_finger_states(hand_landmarks)
        print("Finger states: ", finger_states)
        palm1, palm2 = self.palm_orientation(hand_landmarks)
        """
        Identifies gestures based on which fingers are extended or folded.
        """        
        if palm1:
            ### NOT VERY ROBUST
            if palm2:#cross_vec:
                return "Palm Up"
            elif not palm2:#cross_vec:
                return "Palm Down"
        elif not palm1:
            if finger_states == [1, 1, 1, 1, 1]:
                return "Open Palm"
            else:
                return "Unknown Gesture"

    def get_hand_landmarks(self):
        return self.hand_landmarks

    def get_frame(self):
        return self.frame