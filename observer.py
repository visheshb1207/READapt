# OBSERVE MODULE (CONTINUOUS + FIXED MODE)
# File: observe.py

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from typing import Dict, Optional, Tuple, List
from collections import deque

# CONFIG

DEFAULT_OBSERVE_DURATION = 10

EAR_THRESHOLD = 0.21
EAR_CONSEC_FRAMES = 2

NORMAL_BLINK_RATE = 18
BLINK_RATE_LOW_FATIGUE = 8
BLINK_RATE_HIGH_FATIGUE = 26

GAZE_STABILITY_THRESHOLD = 0.08
HEAD_MOVEMENT_THRESHOLD = 0.08

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

LEFT_EYE_CENTER_IDX = 468
RIGHT_EYE_CENTER_IDX = 473
NOSE_TIP_IDX = 1

ATTENTION_WINDOW_SIZE = 30

mp_face_mesh = mp.solutions.face_mesh

# UTILITY FUNCTIONS

def euclidean_dist(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(eye_points):
    if len(eye_points) != 6:
        return 0.3
    p1, p2, p3, p4, p5, p6 = eye_points
    vert1 = euclidean_dist(p2, p6)
    vert2 = euclidean_dist(p3, p5)
    horiz = euclidean_dist(p1, p4)
    return (vert1 + vert2) / (2.0 * horiz + 1e-6)

def normalize_coord(x, y, w, h):
    cx, cy = w / 2, h / 2
    return (x - cx) / cx, (y - cy) / cy

# ATTENTION TRACKER

class AttentionTracker:
    def __init__(self):
        self.total_frames = 0
        self.face_frames = 0
        self.no_face_frames = 0

        self.blink_count = 0
        self.ear_below_counter = 0

        self.ear_values = []
        self.gaze_positions = []
        self.head_positions = []

        self.focused_streak = 0
        self.max_focused_streak = 0
        self.distraction_events = 0
        self.last_was_focused = True

        self.start_time = time.time()

    def get_elapsed_time(self):
        return time.time() - self.start_time

    def update_focus_streak(self, is_focused):
        if is_focused:
            self.focused_streak += 1
            self.max_focused_streak = max(self.max_focused_streak, self.focused_streak)
        else:
            if self.last_was_focused:
                self.distraction_events += 1
            self.focused_streak = 0
        self.last_was_focused = is_focused

# CORE FRAME PROCESSOR

def process_frame(frame, face_mesh, tracker):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    tracker.total_frames += 1

    if not result.multi_face_landmarks:
        tracker.no_face_frames += 1
        tracker.update_focus_streak(False)
        return False

    landmarks = result.multi_face_landmarks[0].landmark

    def get_points(idxs):
        return [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs]

    left_eye = get_points(LEFT_EYE_IDX)
    right_eye = get_points(RIGHT_EYE_IDX)

    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
    tracker.ear_values.append(ear)

    if ear < EAR_THRESHOLD:
        tracker.ear_below_counter += 1
    else:
        if tracker.ear_below_counter >= EAR_CONSEC_FRAMES:
            tracker.blink_count += 1
        tracker.ear_below_counter = 0

    nose = landmarks[NOSE_TIP_IDX]
    head_dx, head_dy = normalize_coord(nose.x * w, nose.y * h, w, h)
    tracker.head_positions.append((head_dx, head_dy))

    if len(landmarks) > 470:
        left_iris = landmarks[LEFT_EYE_CENTER_IDX]
        right_iris = landmarks[RIGHT_EYE_CENTER_IDX]
        eye_x = (left_iris.x + right_iris.x) / 2 * w
        eye_y = (left_iris.y + right_iris.y) / 2 * h
    else:
        eye_x = (left_eye[0][0] + right_eye[0][0]) / 2
        eye_y = (left_eye[0][1] + right_eye[0][1]) / 2

    gaze_dx, gaze_dy = normalize_coord(eye_x, eye_y, w, h)
    tracker.gaze_positions.append((gaze_dx, gaze_dy))

    gaze_dist = math.sqrt(gaze_dx ** 2 + gaze_dy ** 2)
    head_dist = math.sqrt(head_dx ** 2 + head_dy ** 2)

    is_focused = gaze_dist < 0.4 and head_dist < 0.3
    tracker.update_focus_streak(is_focused)

    tracker.face_frames += 1
    return True

# METRIC COMPUTATION

def compute_focus_and_fatigue(tracker, duration_sec):
    total_frames = tracker.total_frames
    face_frames = tracker.face_frames

    face_ratio = face_frames / max(total_frames, 1)
    blink_rate = (tracker.blink_count / duration_sec) * 60 if duration_sec > 0 else 0

    gaze_std = float(np.mean(np.std(tracker.gaze_positions, axis=0))) if tracker.gaze_positions else 0
    head_std = float(np.mean(np.std(tracker.head_positions, axis=0))) if tracker.head_positions else 0

    focus = 100 * face_ratio
    focus -= min((gaze_std * 400), 25)
    focus -= min((head_std * 400), 25)

    focus_score = max(0, min(100, focus))

    fatigue = "low"
    if blink_rate > BLINK_RATE_HIGH_FATIGUE or focus_score < 40:
        fatigue = "high"
    elif blink_rate > NORMAL_BLINK_RATE or focus_score < 60:
        fatigue = "moderate"

    fps = total_frames / max(duration_sec, 1)
    attention_span = tracker.max_focused_streak / max(fps, 1)

    return {
        "focus_score": round(focus_score, 1),
        "fatigue_level": fatigue,
        "blink_rate_per_min": round(blink_rate, 1),
        "gaze_stability": round(1 / (1 + gaze_std), 3),
        "head_movement": round(head_std, 3),
        "attention_span_seconds": round(attention_span, 1),
        "face_presence_ratio": round(face_ratio, 3),
        "distraction_events": int(tracker.distraction_events),
        "total_frames": int(tracker.total_frames),
        "face_frames": int(tracker.face_frames),
        "no_face_frames": int(tracker.no_face_frames),
        "blink_count": int(tracker.blink_count),
        "avg_ear": round(np.mean(tracker.ear_values), 3) if tracker.ear_values else 0.0,
        "needs_break": fatigue == "high",
        "is_engaged": focus_score > 70 and fatigue == "low"
    }

# FIXED DURATION MODE

def run_observe_module(duration_sec=DEFAULT_OBSERVE_DURATION, show_debug_window=False, camera_index=0):
    tracker = AttentionTracker()
    cap = cv2.VideoCapture(camera_index)

    with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        start = time.time()

        while time.time() - start < duration_sec:
            ret, frame = cap.read()
            if not ret:
                break
            process_frame(frame, face_mesh, tracker)

    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass

    return compute_focus_and_fatigue(tracker, tracker.get_elapsed_time())

# CONTINUOUS MODE FOR STREAMLIT

def start_continuous_observe(camera_index=0, show_debug_window=False):
    tracker = AttentionTracker()
    cap = cv2.VideoCapture(camera_index)

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return cap, face_mesh, tracker


def process_continuous_frame(cap, face_mesh, tracker, show_debug_window=False):
    if cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        return

    process_frame(frame, face_mesh, tracker)

    if show_debug_window:
        cv2.imshow("Observe", frame)
        cv2.waitKey(1)


def stop_continuous_observe(cap, face_mesh, tracker):
    if cap:
        cap.release()
    if face_mesh:
        face_mesh.close()

    try:
        cv2.destroyAllWindows()
    except:
        pass

    return compute_focus_and_fatigue(tracker, tracker.get_elapsed_time())

