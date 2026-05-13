import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class IGFeatureVisualizer:
    def __init__(self, frame_shape):
        self.H, self.W, self.D = frame_shape
        self.center_frame = (self.W // 2, self.H // 2)
        
        # Match ig_hand_state.py FINGERS dictionary exactly
        self.FINGERS = {
            "name": ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"],
            "base_idx": [1, 5, 9, 13, 17],
            "pip_idx": [2, 6, 10, 14, 18],
            "dip_idx": [3, 7, 11, 15, 19],
            "tip_idx": [4, 8, 12, 16, 20]
        }
        self.WRIST_idx = 0

    def to_px(self, lm):
        return int(lm.x * self.W), int(lm.y * self.H)

    def get_vec(self, lm):
        return np.array([lm.x, lm.y, lm.z])

    def vector_angle(self, v1, v2):
        unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
        dot_product = np.dot(unit_v1, unit_v2)
        return np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    
    def draw_angle_arc(self, frame, p1, p2, p3, radius, color, thickness=2):
        """Draws a 2D arc between line p2-p1 and line p2-p3 at vertex p2."""
        angle1 = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        angle2 = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]))

        # Ensure angles are positive
        if angle1 < 0: angle1 += 360
        if angle2 < 0: angle2 += 360

        start_angle = min(angle1, angle2)
        end_angle = max(angle1, angle2)

        # Draw the shortest path arc
        if end_angle - start_angle > 180:
            start_angle = max(angle1, angle2)
            end_angle = min(angle1, angle2) + 360

        cv2.ellipse(frame, p2, (radius, radius), 0, start_angle, end_angle, color, thickness)

    # ==========================================
    # RULE 6: Position (Key 1)
    # ==========================================
    def draw_position(self, frame, landmarks):
        wrist_px = self.to_px(landmarks[self.WRIST_idx])
        
        # 1. Geometric Center of Hand
        all_px = np.array([self.to_px(lm) for lm in landmarks])
        geo_center = np.mean(all_px, axis=0).astype(int)
        cv2.circle(frame, tuple(geo_center), 8, (255, 255, 255), -1) # White geometric center
        
        # 2. Frame Center and Margins (20%)
        cv2.circle(frame, self.center_frame, 5, (255, 0, 255), -1)
        margin_w, margin_h = int(self.W * 0.20), int(self.H * 0.20)
        top_left = (self.center_frame[0] - margin_w, self.center_frame[1] - margin_h)
        bottom_right = (self.center_frame[0] + margin_w, self.center_frame[1] + margin_h)
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2) # Semantic bounding box

        # 3. Displacement Lines (dx, dy)
        dx = self.center_frame[0] - wrist_px[0]
        dy = self.center_frame[1] - wrist_px[1]
        
        intersect = (self.center_frame[0], wrist_px[1])
        cv2.line(frame, self.center_frame, intersect, (0, 0, 255), 2) # dy
        cv2.line(frame, intersect, wrist_px, (255, 0, 0), 2) # dx
        
        # Display text cleanly on the side
        cv2.putText(frame, "--- Position ---", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"dx (Blue): {dx}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"dy (Red): {dy}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ==========================================
    # RULE 5: Orientation (Key 2)
    # ==========================================
    def draw_orientation(self, frame, landmarks, label):
        wrist_px = self.to_px(landmarks[self.WRIST_idx])
        idx_base_px = self.to_px(landmarks[self.FINGERS["base_idx"][1]])
        pinky_base_px = self.to_px(landmarks[self.FINGERS["base_idx"][4]])
        mid_base_px = self.to_px(landmarks[self.FINGERS["base_idx"][2]])

        # Draw Base Vectors
        cv2.line(frame, pinky_base_px, idx_base_px, (255, 255, 0), 3) # vec1 (Cyan)
        cv2.line(frame, wrist_px, mid_base_px, (255, 255, 0), 3) # vec2 (Cyan)

        # Cross Product Logic
        vec1 = self.get_vec(landmarks[self.FINGERS["base_idx"][4]]) - self.get_vec(landmarks[self.FINGERS["base_idx"][1]])
        vec2 = self.get_vec(landmarks[self.WRIST_idx]) - self.get_vec(landmarks[self.FINGERS["base_idx"][2]])

        if label == 'Right':
            palm_normal = np.cross(vec1, vec2)
        else:
            palm_normal = np.cross(vec2, vec1)
            
        palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-6)

        # Draw Normal
        end_px = (int(wrist_px[0] + palm_normal[0] * -100), int(wrist_px[1] + palm_normal[1] * -100))
        cv2.line(frame, wrist_px, end_px, (0, 0, 255), 4)

    # ==========================================
    # RULE 1a: Thumb Flexion (Key 3)
    # ==========================================
    def draw_thumb_flexion(self, frame, landmarks):
        tip_idx = self.FINGERS["tip_idx"][0]
        pip_idx = self.FINGERS["pip_idx"][0]
        p_base_idx = self.FINGERS["base_idx"][4]

        tip_px = self.to_px(landmarks[tip_idx])
        pip_px = self.to_px(landmarks[pip_idx])
        p_base_px = self.to_px(landmarks[p_base_idx])

        # Vectors
        v_tip = self.get_vec(landmarks[tip_idx])
        v_pip = self.get_vec(landmarks[pip_idx])
        v_pbase = self.get_vec(landmarks[p_base_idx])

        dist_thumb_palm = np.linalg.norm(v_tip - v_pbase)
        dist_pip_palm = np.linalg.norm(v_pip - v_pbase)

        # Draw Lines
        cv2.line(frame, tip_px, p_base_px, (0, 255, 0), 2)
        cv2.line(frame, pip_px, p_base_px, (0, 0, 255), 2)

        state = "EXTENDED" if dist_thumb_palm > dist_pip_palm else "FOLDED"

        # Display text cleanly on the side
        cv2.putText(frame, f"Thumb: {state}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Tip->Palm (Green): {dist_thumb_palm:.3f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"PIP->Palm (Red): {dist_pip_palm:.3f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ==========================================
    # RULE 1b: Other Finger Flexion (Keys 4, 5, 6, 7)
    # ==========================================
    def draw_finger_flexion(self, frame, landmarks, f_idx):
        f_name = self.FINGERS["name"][f_idx]
        base = self.FINGERS["base_idx"][f_idx]
        pip = self.FINGERS["pip_idx"][f_idx]
        dip = self.FINGERS["dip_idx"][f_idx]
        tip = self.FINGERS["tip_idx"][f_idx]
        wrist = self.WRIST_idx

        # 3D Vectors
        v1 = self.get_vec(landmarks[pip]) - self.get_vec(landmarks[base])
        v2 = self.get_vec(landmarks[dip]) - self.get_vec(landmarks[pip])
        v3 = self.get_vec(landmarks[tip]) - self.get_vec(landmarks[dip])

        angle1 = self.vector_angle(v1, v2)
        angle2 = self.vector_angle(v2, v3)
        total_curl = angle1 + angle2

        # Pixel coordinates for drawing
        base_px = self.to_px(landmarks[base])
        pip_px = self.to_px(landmarks[pip])
        dip_px = self.to_px(landmarks[dip])
        tip_px = self.to_px(landmarks[tip])

        # Drawing the bones
        cv2.line(frame, base_px, pip_px, (0, 255, 255), 3)
        cv2.line(frame, pip_px, dip_px, (0, 255, 255), 3)
        cv2.line(frame, dip_px, tip_px, (0, 255, 255), 3)

        # # --- DRAWING THE ARCS ---
        # # Arc 1: Angle at PIP joint -> Green
        # self.draw_angle_arc(frame, base_px, pip_px, dip_px, radius=20, color=(0, 255, 0), thickness=2)
        
        # # Arc 2: Angle at DIP joint -> Blue
        # self.draw_angle_arc(frame, pip_px, dip_px, tip_px, radius=20, color=(255, 0, 0), thickness=2)

        # # Arc 3: Total Angle (Virtual representation) -> Orange
        # # We translate the DIP-Tip vector to start at the PIP joint to visualize the total bend sum
        # v3_2d_x = tip_px[0] - dip_px[0]
        # v3_2d_y = tip_px[1] - dip_px[1]
        # virtual_tip_px = (pip_px[0] + v3_2d_x, pip_px[1] + v3_2d_y)
        
        # # Draw a thin reference line for the virtual vector
        # cv2.line(frame, pip_px, virtual_tip_px, (0, 165, 255), 1) 
        # # Draw the larger total arc
        # self.draw_angle_arc(frame, base_px, pip_px, virtual_tip_px, radius=35, color=(0, 165, 255), thickness=3)

        # Drawing
        cv2.line(frame, self.to_px(landmarks[base]), self.to_px(landmarks[pip]), (0, 255, 255), 3)
        cv2.line(frame, self.to_px(landmarks[pip]), self.to_px(landmarks[dip]), (0, 255, 255), 3)
        cv2.line(frame, self.to_px(landmarks[dip]), self.to_px(landmarks[tip]), (0, 255, 255), 3)


        # Distance backup check
        tip_w_dist = np.linalg.norm(self.get_vec(landmarks[tip]) - self.get_vec(landmarks[wrist]))
        pip_w_dist = np.linalg.norm(self.get_vec(landmarks[pip]) - self.get_vec(landmarks[wrist]))
        folded_by_dist = tip_w_dist < pip_w_dist
        
        # # Display text cleanly on the side
        # cv2.putText(frame, f"{f_name} Curl: {int(total_curl)} | Dist Fold: {folded_by_dist}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.putText(frame, f"Angle 1: {int(angle1)} deg", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv2.putText(frame, f"Angle 2: {int(angle2)} deg", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display text cleanly on the side
        cv2.putText(frame, f"--- {f_name} Flexion ---", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Angle 1 (Green): {int(angle1)} deg", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Angle 2 (Blue): {int(angle2)} deg", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Total Curl (Orange): {int(total_curl)} deg", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, f"Dist Folded Backup: {folded_by_dist}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    # ==========================================
    # RULE 3: Contact (Key 8)
    # ==========================================
    def draw_contact(self, frame, landmarks):
        thumb_tip = self.get_vec(landmarks[self.FINGERS["tip_idx"][0]])
        thumb_px = self.to_px(landmarks[self.FINGERS["tip_idx"][0]])
        
        cv2.putText(frame, "--- Finger Contact ---", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset = 110

        # Create a flag to track if the thumb is touching ANY finger
        thumb_is_touching = False
        circle = 1

        for i in range(1, 5):
            f_name = self.FINGERS["name"][i]
            f_tip = self.get_vec(landmarks[self.FINGERS["tip_idx"][i]])
            f_px = self.to_px(landmarks[self.FINGERS["tip_idx"][i]])
            
            dist = np.linalg.norm(thumb_tip - f_tip)
            
            # Check contact for the current finger
            # if dist <= 0.045:
            if dist <= 0.1:
                color = (0, 255, 0)
                circle = -1
                thumb_is_touching = True # Trigger the flag!
            else:
                color = (0, 0, 255)

            # Draw just the finger tips
            cv2.circle(frame, f_px, 5, color, circle)
            
            # Draw the connection line
            cv2.line(frame, thumb_px, f_px, color, 2)
            
            # Display text cleanly on the side
            cv2.putText(frame, f"Thumb -> {f_name}: {dist:.3f}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30

        # Draw the thumb tip ONCE after the loop has finished evaluating all fingers
        thumb_color = (0, 255, 0) if thumb_is_touching else (0, 0, 255)
        cv2.circle(frame, thumb_px, 5, thumb_color, circle)


def main():
    cap = cv2.VideoCapture(index=0)
    viz_mode = 1 
    
    # Temporal history buffer (window_size=15)
    wrist_history = deque(maxlen=15)

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1) 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    
                    # Keep temporal history updated
                    vis = IGFeatureVisualizer(frame.shape)
                    wrist_px = vis.to_px(hand_landmarks.landmark[0])
                    wrist_history.append(wrist_px)

                    # Custom, smaller drawing specs & connection lines for the landmarks
                    # Grab the default colorful dictionary
                    custom_rainbow_spec = mp_drawing_styles.get_default_hand_landmarks_style()
                    custom_rainbow_connection = mp_drawing_styles.get_default_hand_connections_style()

                    for landmark in custom_rainbow_spec:
                        custom_rainbow_spec[landmark].thickness = 2
                        custom_rainbow_spec[landmark].circle_radius = 1

                    for connection in custom_rainbow_connection:
                        custom_rainbow_connection[connection].thickness = 2
                    

                    # Only draw the full skeleton if we aren't focusing on specific fingers
                    if viz_mode not in [3, 4, 5, 6, 7, 8]:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  landmark_drawing_spec=custom_rainbow_spec,
                                                  connection_drawing_spec=custom_rainbow_connection)

                    label = handedness.classification[0].label

                    # Router
                    if viz_mode == 1: vis.draw_position(frame, hand_landmarks.landmark)
                    elif viz_mode == 2: vis.draw_orientation(frame, hand_landmarks.landmark, label)
                    elif viz_mode == 3: vis.draw_thumb_flexion(frame, hand_landmarks.landmark)
                    elif viz_mode == 4: vis.draw_finger_flexion(frame, hand_landmarks.landmark, 1)
                    elif viz_mode == 5: vis.draw_finger_flexion(frame, hand_landmarks.landmark, 2)
                    elif viz_mode == 6: vis.draw_finger_flexion(frame, hand_landmarks.landmark, 3)
                    elif viz_mode == 7: vis.draw_finger_flexion(frame, hand_landmarks.landmark, 4)
                    elif viz_mode == 8: vis.draw_contact(frame, hand_landmarks.landmark)
                    elif viz_mode == 9:
                        # Draw Temporal Path (ig_temporal_gesture logic)
                        if len(wrist_history) > 1:
                            pts = np.array(wrist_history, np.int32).reshape((-1, 1, 2))
                            cv2.polylines(frame, [pts], False, (0, 255, 255), 3)
                            cv2.circle(frame, wrist_history[-1], 6, (0, 0, 255), -1) # Current pos
                            cv2.putText(frame, f"Trajectory (15 frames)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("IG Feature Extractions", frame)
            
            # Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            elif ord("0") <= key <= ord("9"): viz_mode = int(chr(key))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("1:Pos | 2:Orient | 3:Thumb | 4:Idx | 5:Mid | 6:Rng | 7:Pnk | 8:Cont | 9:Path | 0:Off")
    main()