import os
import time
import math
import json
import base64
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import cv2
import onnxruntime as ort
import mediapipe as mp

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173").split(",")
PORT = int(os.getenv("PORT", "8002"))

# UPDATED: Default to the EMA ONNX model
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "resnet18_gazevec_ema.onnx")

# Config
MAX_BUFFER = 120
GAZE_SMOOTHING_WINDOW = 5
EAR_THRESHOLD = 0.25
GAZE_THRESHOLD_DEG = 15.0
FRAME_SAMPLING_INTERVAL_S = 0.1 

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_INPUT_SIZE = (224, 224)
PITCH_CLAMP = (-60.0, 60.0)
YAW_CLAMP = (-60.0, 60.0)

# -----------------------------
# Pydantic models
# -----------------------------
class ImageAnalysisRequest(BaseModel):
    image_data: str
    session_id: str

class AnswerEvaluationRequest(BaseModel):
    question: str
    answer: str
    job_role: str = "Software Engineer"

# -----------------------------
# Session & State
# -----------------------------
@dataclass
class BehaviorState:
    ear_series: deque
    gaze_history: deque
    blink_count: int = 0
    last_blink_time: float = 0.0
    total_frames: int = 0
    bg_frames_detected_count: int = 0
    stress_frames: int = 0
    smile_frames: int = 0
    gaze_metrics: dict = None
    last_sample_time: float = 0.0
    # Smoothing State
    last_pitch: float = 0.0
    last_yaw: float = 0.0
    # Time-windowed tracking
    blink_timestamps: deque = None  # Track when blinks occurred
    frame_timestamps: deque = None  # Track frame arrival times
    engagement_window: deque = None  # Track recent engagement (1=engaged, 0=not)
    emotion_history: deque = None  # Track recent emotions for smoothing
    head_pose_history: deque = None # To calculate head stability/fidgeting
    confidence_history: deque = None # To track behavior-based human confidence

    def __post_init__(self):
        if self.blink_timestamps is None:
            self.blink_timestamps = deque(maxlen=100)
        if self.frame_timestamps is None:
            self.frame_timestamps = deque(maxlen=1800)
        if self.engagement_window is None:
            self.engagement_window = deque(maxlen=1800)
        if self.emotion_history is None:
            self.emotion_history = deque(maxlen=30)
        if self.head_pose_history is None:
            self.head_pose_history = deque(maxlen=60) # ~6 seconds of stability data
        if self.confidence_history is None:
            self.confidence_history = deque(maxlen=60)
        if self.gaze_metrics is None:
            self.gaze_metrics = {
                "avg_gaze_deviation": 0.0,
                "eye_contact_ratio": 0.0,
                "blink_rate": 0.0,
                "engagement_score": 0.0,
                "stress_level": "Low",
                "dominant_emotion": "Neutral",
                "confidence": 0.0 # Standardized: Human behavioral confidence
            }

sessions: Dict[str, BehaviorState] = {}

# -----------------------------
# Globals
# -----------------------------
ort_sess: Optional[ort.InferenceSession] = None
mp_face_mesh = None
mp_face_detection = None

# -----------------------------
# Vision Utilities
# -----------------------------
def safe_clip(val: float, minv: float, maxv: float) -> float:
    return max(minv, min(maxv, val))

def clamp_angles(pitch: float, yaw: float) -> Tuple[float, float]:
    return (safe_clip(pitch, PITCH_CLAMP[0], PITCH_CLAMP[1]),
            safe_clip(yaw, YAW_CLAMP[0], YAW_CLAMP[1]))

def decode_base64_image(data_uri_or_b64: str):
    try:
        if "base64," in data_uri_or_b64:
            data = data_uri_or_b64.split("base64,")[1]
        else:
            data = data_uri_or_b64
        b = base64.b64decode(data)
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Image Decode Error: {e}")
        return None

def preprocess_face_for_model(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        # 1. Convert BGR to RGB (Crucial for ResNet trained on RGB)
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Resize to 224x224
        face_resized = cv2.resize(face_rgb, (MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]))
        
        # 3. Normalize [0,1] and then ImageNet Mean/Std
        face_float = face_resized.astype(np.float32) / 255.0
        face_norm = (face_float - IMAGENET_MEAN) / IMAGENET_STD
        
        # 4. Transpose to CHW and Add Batch Dimension -> (1, 3, 224, 224)
        return np.expand_dims(np.transpose(face_norm, (2,0,1)).astype(np.float32), axis=0)
    except Exception:
        return None

def gaze_vector_to_pitch_yaw_deg(vec: np.ndarray) -> Tuple[float, float]:
    """
    Converts 3D gaze vector to Pitch and Yaw degrees.
    Matches the training script logic:
       yaw = atan2(-x, -z)
       pitch = asin(y)
    """
    try:
        x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
        
        # UPDATED MATH to match training script
        # Training: pitch = asin(y), yaw = atan2(-x, -z)
        pitch_rad = math.asin(np.clip(y, -1.0, 1.0))
        yaw_rad = math.atan2(-x, -z)
        
        return clamp_angles(math.degrees(pitch_rad), math.degrees(yaw_rad))
    except Exception:
        return 0.0, 0.0

# -----------------------------
# Initialization & App
# -----------------------------
def init_models():
    global ort_sess, mp_face_mesh, mp_face_detection
    
    # 1. ONNX
    if os.path.exists(ONNX_MODEL_PATH):
        try:
            ort_sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
            print(f"✅ Loaded ONNX model from {ONNX_MODEL_PATH}")
            # Warmup
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            ort_sess.run(None, {'input': dummy_input})
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            ort_sess = None
    else:
        print(f"⚠️ ONNX Model not found at {ONNX_MODEL_PATH}. Gaze will be mocked.")

    # 2. MediaPipe
    try:
        import mediapipe.python.solutions as solutions
        mp_face_mesh = solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        mp_face_detection = solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        print("✅ MediaPipe Solutions initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize MediaPipe: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models()
    yield
    if mp_face_mesh: mp_face_mesh.close()
    if mp_face_detection: mp_face_detection.close()

app = FastAPI(title="Gaze API", version="2.0", lifespan=lifespan)

@app.get("/health")
async def health():
    print(f"[GET] /health - Checking system status")
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "onnx_model_loaded": ort_sess is not None,
        "mediapipe_loaded": mp_face_mesh is not None
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_head_pose_mediapipe(landmarks, w, h):
    """Calculate head pose (pitch, yaw, roll) from MediaPipe landmarks"""
    # 3D model points (generic face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (1)
        (0.0, -330.0, -65.0),        # Chin (152)
        (-225.0, 170.0, -135.0),     # Left eye left corner (33)
        (225.0, 170.0, -135.0),      # Right eye right corner (263)
        (-150.0, -150.0, -125.0),    # Left mouth corner (61)
        (150.0, -150.0, -125.0)      # Right mouth corner (291)
    ])
    
    # 2D image points from landmarks
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),      # Nose tip
        (landmarks[152].x * w, landmarks[152].y * h),  # Chin
        (landmarks[33].x * w, landmarks[33].y * h),    # Left eye
        (landmarks[263].x * w, landmarks[263].y * h),  # Right eye
        (landmarks[61].x * w, landmarks[61].y * h),    # Left mouth
        (landmarks[291].x * w, landmarks[291].y * h)   # Right mouth
    ], dtype="double")
    
    # Camera internals
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4,1))
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0.0, 0.0, 0.0
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles
    sy = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
    singular = sy < 1e-6
    
    if not singular:
        pitch = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
        yaw = math.atan2(-rotation_matrix[2,0], sy)
        roll = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
    else:
        pitch = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
        yaw = math.atan2(-rotation_matrix[2,0], sy)
        roll = 0
    
    # NEW: Normalize pitch to [-180, 180] and then wrap around 0
    # Many PnP solvers return 180-centered pitch for front-facing cams
    p_deg = math.degrees(pitch)
    if p_deg > 90: p_deg -= 180
    elif p_deg < -90: p_deg += 180
        
    return p_deg, math.degrees(yaw), math.degrees(roll)

def analyze_gaze_real(frame, face_rect):
    h, w, _ = frame.shape
    x, y, fw, fh = face_rect
    
    # Expand box slightly
    center_x, center_y = x + fw // 2, y + fh // 2
    max_dim = max(fw, fh) * 1.2 
    x1 = int(max(0, center_x - max_dim // 2))
    y1 = int(max(0, center_y - max_dim // 2))
    x2 = int(min(w, center_x + max_dim // 2))
    y2 = int(min(h, center_y + max_dim // 2))
    
    face_roi = frame[y1:y2, x1:x2]
    pitch, yaw = 0.0, 0.0
    
    if ort_sess and face_roi.size > 0:
        model_input = preprocess_face_for_model(face_roi)
        if model_input is not None:
            # Inference
            output = ort_sess.run(None, {'input': model_input})[0][0]
            pitch, yaw = gaze_vector_to_pitch_yaw_deg(output)
            
    return pitch, yaw


def calculate_ear(landmarks, indices):
    v1 = np.linalg.norm(np.array([landmarks[indices[1]].x, landmarks[indices[1]].y]) - 
                        np.array([landmarks[indices[5]].x, landmarks[indices[5]].y]))
    v2 = np.linalg.norm(np.array([landmarks[indices[2]].x, landmarks[indices[2]].y]) - 
                        np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]))
    h = np.linalg.norm(np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) - 
                       np.array([landmarks[indices[3]].x, landmarks[indices[3]].y]))
    ear = (v1 + v2) / (2.0 * h + 1e-6)
    return ear

def calculate_eyebrow_raise(landmarks) -> float:
    """Scale-invariant eyebrow raise detection using iris distance as unit"""
    # Unit: Distance between eye centers (468: Left Iris, 473: Right Iris)
    p_left_eye = np.array([landmarks[468].x, landmarks[468].y])
    p_right_eye = np.array([landmarks[473].x, landmarks[473].y])
    face_scale = np.linalg.norm(p_left_eye - p_right_eye) + 1e-6
    
    # Left brow (70), Right brow (300)
    avg_brow_y = (landmarks[70].y + landmarks[300].y) / 2.0
    avg_eye_y = (p_left_eye[1] + p_right_eye[1]) / 2.0
    
    # vertical_gap increases when brows go UP (Y decreases)
    vertical_gap = (avg_eye_y - avg_brow_y) / face_scale
    
    # Calibrate: Neutral gap is ~0.40, Raised is > 0.55
    # Adjusted range [0.42, 0.62] -> [0.0, 1.0] for better robustness
    normalized = np.clip((vertical_gap - 0.42) / 0.20, 0.0, 1.0)
    return normalized

def calculate_mouth_aspect_ratio(landmarks) -> float:
    top = np.array([landmarks[13].x, landmarks[13].y])
    bot = np.array([landmarks[14].x, landmarks[14].y])
    left = np.array([landmarks[78].x, landmarks[78].y])
    right = np.array([landmarks[308].x, landmarks[308].y])
    
    mouth_h = np.linalg.norm(top - bot)
    mouth_w = np.linalg.norm(left - right)
    
    mar = mouth_h / (mouth_w + 0.001)
    return mar

def calculate_mouth_curvature(landmarks) -> float:
    """Calculate mouth corner curvature relative to lip center (13, 14)"""
    # Corners: 61 (L), 291 (R)
    # Center: 13 (Upper lip top), 14 (Lower lip bottom)
    left_corner = landmarks[61].y
    right_corner = landmarks[291].y
    mouth_center_y = (landmarks[13].y + landmarks[14].y) / 2.0
    
    avg_corner_y = (left_corner + right_corner) / 2.0
    # Smile makes corners go UP (smaller Y coordinate), so mouth_center_y - avg_corner_y becomes positive
    # Neutral: avg_corner_y should be close to mouth_center_y
    curvature = mouth_center_y - avg_corner_y
    
    # Scale for detection thresholding
    return np.clip(curvature * 150, -1.0, 1.0)

def analyze_emotion_v2(landmarks):
    """Enhanced emotion detection with adjusted thresholds and returned metrics for debug"""
    eyebrow_raise = calculate_eyebrow_raise(landmarks)
    mouth_ratio = calculate_mouth_aspect_ratio(landmarks)
    mouth_curve = calculate_mouth_curvature(landmarks)
    
    emotion = "Neutral"
    # Adjusted: Lowered thresholds to make detection more sensitive
    if eyebrow_raise > 0.4 or mouth_ratio > 0.15: 
        emotion = "Surprised"
    elif mouth_curve > 0.12: # Even a slight smile is now Happy
        emotion = "Happy"
        
    return emotion, (eyebrow_raise, mouth_ratio, mouth_curve)

def get_dominant_emotion(state: 'BehaviorState') -> str:
    if len(state.emotion_history) < 5:
        return "Neutral"
    emotion_counts = {}
    for emotion in state.emotion_history:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    return max(emotion_counts, key=emotion_counts.get)

def analyze_stress_level(state: 'BehaviorState', current_time: float) -> str:
    if state.total_frames < 10: return "Low"
    
    WINDOW_SECONDS = 60.0
    window_start = current_time - WINDOW_SECONDS
    
    recent_blinks = sum(1 for ts in state.blink_timestamps if ts >= window_start)
    blinks_per_minute = recent_blinks 
    
    recent_frames = sum(1 for ts in state.frame_timestamps if ts >= window_start)
    if recent_frames < 10: return "Low"
    
    recent_engagement_data = [state.engagement_window[i] for i in range(len(state.engagement_window)) 
                              if i < len(state.frame_timestamps) and state.frame_timestamps[i] >= window_start]
    engagement = sum(recent_engagement_data) / len(recent_engagement_data) if recent_engagement_data else 1.0
    
    avg_deviation = state.gaze_metrics.get("avg_gaze_deviation", 0.0)
    
    stress_score = 0
    if blinks_per_minute > 30: stress_score += 40
    elif blinks_per_minute > 20: stress_score += 20
    
    if avg_deviation > 25: stress_score += 30
    elif avg_deviation > 15: stress_score += 15
    
    if engagement < 0.5: stress_score += 30
    elif engagement < 0.7: stress_score += 15
    
    # Calculate Gaze Jitter/Deviation (Dynamic Stress Indicator)
    # If the person is looking away or eyes are moving rapidly
    if len(state.gaze_history) > 10:
        recent_gaze = list(state.gaze_history)[-60:] # last 6 seconds
        pitches = [g[0] for g in recent_gaze]
        yaws = [g[1] for g in recent_gaze]
        avg_deviation = (np.std(pitches) + np.std(yaws)) / 2.0
        state.gaze_metrics["avg_gaze_deviation"] = float(avg_deviation)
        
        if avg_deviation > 25: stress_score += 30
        elif avg_deviation > 15: stress_score += 15

    if stress_score >= 60: return "High"
    elif stress_score >= 30: return "Medium"
    else: return "Low"

def calculate_human_confidence(state: 'BehaviorState') -> float:
    """
    Calculates a 0-100 Confidence Score based on behavioral cues:
    1. Eye Contact (35%)
    2. Head Stability (25%)
    3. Blink Rate (20%)
    4. Emotion/Presence (20%)
    """
    # 1. Eye Contact (Engagement window score)
    # Uses the last 300 frames (~30 seconds at 10fps)
    recent_engagement = list(state.engagement_window)[-300:]
    eye_score = (sum(recent_engagement) / len(recent_engagement) * 100) if recent_engagement else 100
    
    # 2. Head Stability (Standard deviation of pitch/yaw)
    # Confident people have steady heads; nervous people fidget.
    recent_poses = list(state.head_pose_history)[-60:] # last 6 seconds
    if len(recent_poses) > 10:
        pitches = [p[0] for p in recent_poses]
        yaws = [p[1] for p in recent_poses]
        stability = (np.std(pitches) + np.std(yaws)) / 2.0
        # Lower stability (fidgeting) reduces score. 0-5 degrees is steady.
        stability_score = max(0, min(100, 100 - (stability * 10))) 
    else:
        stability_score = 100

    # 3. Blink Rate Score (Target 10-20 blinks per minute)
    # High blinks (>30) or Zero blinks (frozen) reduced score.
    now = time.time()
    recent_blinks = sum(1 for ts in state.blink_timestamps if ts > (now - 60))
    if recent_blinks > 30: 
        blink_score = max(0, 100 - (recent_blinks - 30) * 5)
    elif recent_blinks < 2:
        blink_score = 50 # Unnatural/Stiff
    else:
        blink_score = 100

    # 4. Emotion/Vibe Score
    # Neutral/Happy/Surprised only
    emotion = state.gaze_metrics.get("dominant_emotion", "Neutral")
    emotion_map = {"Happy": 100, "Neutral": 90, "Surprised": 85} # High score for active engagement
    vibe_score = emotion_map.get(emotion, 90)

    # Weighted Average
    final_score = (
        (eye_score * 0.35) + 
        (stability_score * 0.25) + 
        (blink_score * 0.20) + 
        (vibe_score * 0.20)
    )
    
    # Scale to 0.0 - 1.0 range (Frontend multiplies by 100 to show %)
    return round(float(final_score) / 100.0, 3)

# -----------------------------
# HTTP Analysis Endpoint
# -----------------------------
@app.post("/api/analyze-frame")
async def analyze_frame_endpoint(req: ImageAnalysisRequest):
    session_id = req.session_id
    if session_id not in sessions:
        sessions[session_id] = BehaviorState(deque(maxlen=MAX_BUFFER), deque(maxlen=MAX_BUFFER))
    
    state = sessions[session_id]
    print(f"[POST] /api/analyze-frame - Session: {session_id}")
    
    frame = decode_base64_image(req.image_data)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
        
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mesh_results = mp_face_mesh.process(rgb_frame)
    
    face_detected = False
    pitch, yaw = 0.0, 0.0
    detection_score = 0.0
    
    if mesh_results.multi_face_landmarks:
        face_detected = True
        lms = mesh_results.multi_face_landmarks[0].landmark
        
        # 1. Detection (Needed for ONNX bbox)
        detection_results = mp_face_detection.process(rgb_frame)
        if detection_results.detections:
            d = detection_results.detections[0]
            detection_score = d.score[0]  # Get confidence score
            box = d.location_data.relative_bounding_box
            x, y, fw, fh = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
            
            # =========================================================================
            # [OPTION 1]: ResNet Eye Gaze Model (ACTIVE)
            # Uses the custom ONNX model to detect eye gaze direction.
            # =========================================================================
            # raw_pitch, raw_yaw = analyze_gaze_real(frame, (x, y, fw, fh))
            
            # =========================================================================
            # [OPTION 2]: MediaPipe Head Pose (COMMENTED OUT)
            # Uses facial landmarks to estimate head orientation.
            # To use this instead of ResNet, comment out [OPTION 1] and uncomment below.
            # =========================================================================
            raw_pitch, raw_yaw, _ = get_head_pose_mediapipe(lms, w, h)
            raw_pitch = max(-60, min(60, raw_pitch))  # Clamp for safety
            raw_yaw = max(-60, min(60, raw_yaw))
            # =========================================================================
            
            # Smoothing (Exponential Moving Average)
            alpha = 0.2 
            pitch = alpha * raw_pitch + (1 - alpha) * state.last_pitch
            yaw = alpha * raw_yaw + (1 - alpha) * state.last_yaw
            
            state.last_pitch = pitch
            state.last_yaw = yaw
            
            # Store head pose for stability tracking
            state.head_pose_history.append((raw_pitch, raw_yaw))
        
        # 2. Blink Detection (EAR)
        left_ear = calculate_ear(lms, [33, 160, 158, 133, 153, 144])
        right_ear = calculate_ear(lms, [362, 385, 387, 263, 373, 380])
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < 0.25:
            current_time = time.time()
            if current_time - state.last_blink_time > 0.2: 
                state.blink_count += 1
                state.blink_timestamps.append(current_time)
                state.last_blink_time = current_time
        
        # Track timestamp and engagement
        current_time = time.time()
        state.frame_timestamps.append(current_time)
        
        # Get head pose for reference
        head_pitch, head_yaw, head_roll = get_head_pose_mediapipe(lms, w, h)
        
        # Engagement based on BOTH eye gaze AND head pose
        # Threshold balanced to 40° (Wide enough for monitors, tight enough for detection)
        eye_engaged = abs(pitch) < 40 and abs(yaw) < 40 
        head_engaged = abs(head_pitch) < 40 and abs(head_yaw) < 40
        is_engaged = 1 if (eye_engaged and head_engaged) else 0
        state.engagement_window.append(is_engaged)
        
        # DEBUG PRINTS to see why it fails
        if not is_engaged:
            print(f"  [DEBUG] Failed Engagement: Eye(P:{pitch:.1f}, Y:{yaw:.1f}) Head(P:{head_pitch:.1f}, Y:{head_yaw:.1f})")
        
        # 3. Emotion
        current_emotion, raw_metrics = analyze_emotion_v2(lms)
        eb_raise, m_ratio, m_curve = raw_metrics
        state.emotion_history.append(current_emotion)
        state.gaze_history.append((pitch, yaw)) # Update Gaze History
        state.gaze_metrics["dominant_emotion"] = get_dominant_emotion(state)
        
        # 4. Stress & Confidence
        state.gaze_metrics["stress_level"] = analyze_stress_level(state, time.time())
        state.gaze_metrics["confidence"] = calculate_human_confidence(state)
        
        if is_engaged:
             state.bg_frames_detected_count += 1
             
        # Metrics Calculation: Use Rolling Window for Instant Feedback
        # Sum of last N engagement states / N
        recent_engagement = sum(state.engagement_window)
        total_window = len(state.engagement_window)
        
        state.total_frames += 1
        # Use windowed score if we have data, else cumulative
        if total_window > 0:
            state.gaze_metrics["engagement_score"] = float(recent_engagement) / float(total_window)
        else:
            state.gaze_metrics["engagement_score"] = 0.0
            
        state.gaze_metrics["blink_rate"] = state.blink_count 
        
        # Store head pose for output
        state.gaze_metrics["head_pitch"] = head_pitch
        state.gaze_metrics["head_yaw"] = head_yaw
        state.gaze_metrics["head_roll"] = head_roll
        
    # Terminal Output for Debugging
    print(f"\n{'='*70}")
    print(f"Frame Analysis | Session: {session_id} | {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print(f"{'='*70}")
    print(f"Face Detected: {'✅ YES' if face_detected else '❌ NO'} (Det-Score: {detection_score:.2%})")
    if face_detected:
        print(f"Eye Gaze (ResNet Model):")
        print(f"  └─ Pitch: {pitch:+7.2f}° | Yaw: {yaw:+7.2f}°")
        
        head_pitch = state.gaze_metrics.get('head_pitch', 0)
        head_yaw = state.gaze_metrics.get('head_yaw', 0)
        head_roll = state.gaze_metrics.get('head_roll', 0)
        print(f"Head Pose (MediaPipe):")
        print(f"  └─ Pitch: {head_pitch:+7.2f}° | Yaw: {head_yaw:+7.2f}° | Roll: {head_roll:+7.2f}°")
        
        print(f"Behavior:")
        print(f"  └─ Emotion: {state.gaze_metrics['dominant_emotion']} (E-Raise: {eb_raise:.2f}, M-Ratio: {m_ratio:.2f}, M-Curve: {m_curve:.2f})") 
        print(f"  └─ Stress Level: {state.gaze_metrics['stress_level']}") 
        print(f"  └─ Engagement: {state.gaze_metrics['engagement_score']:.1%} (Status: {'ACTIVE' if is_engaged else 'LOOKING-AWAY'})") 
        print(f"  └─ Human Confidence: {state.gaze_metrics['confidence']*100:.1f}%")
        print(f"  └─ Blinks: {state.gaze_metrics['blink_rate']}")
    print(f"{'='*70}\n")
    
    return {
        "type": "analysis_result",
        "session_id": session_id,
        "gaze_analysis": {"pitch": float(pitch), "yaw": float(yaw)} if face_detected else None,
        "behavior_analysis": state.gaze_metrics if face_detected else None,
        "face_detected": face_detected,
        "confidence": state.gaze_metrics["confidence"] if face_detected else None,
        "stress_level": state.gaze_metrics["stress_level"] if face_detected else None,
        "emotion": state.gaze_metrics["dominant_emotion"] if face_detected else None,
        "engagement": state.gaze_metrics["engagement_score"] if face_detected else None,
        "timestamp": datetime.now().isoformat()
    }

# -----------------------------
# Summary & AI Summary Endpoints
# -----------------------------
class BehaviorEvaluationRequest(BaseModel):
    session_id: str
    job_role: str = "Software Engineer"

@app.post("/api/evaluate-behavior")
async def evaluate_behavior_endpoint(req: BehaviorEvaluationRequest):
    print(f"[POST] /api/evaluate-behavior - Session: {req.session_id}")
    if req.session_id not in sessions:
         raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[req.session_id]
    metrics = state.gaze_metrics

    prompt = f"""
    You are a professional behavioral psychologist and technical recruiter. 
    Analyze the following interview behavioral metrics for a candidate applying for the role of {req.job_role}:
    
    Metrics:
    - Eye Contact Ratio (Engagement): {metrics.get('engagement_score', 0):.2f}
    - Blink Rate (indicates stress/focus): {metrics.get('blink_rate', 0)} blinks total
    - Stress Level Detected: {metrics.get('stress_level', 'Unknown')}
    - Dominant Emotion: {metrics.get('dominant_emotion', 'Neutral')}
    
    Provide a professional summary (max 100 words) for the interviewer. 
    Focus on:
    1. Level of confidence and engagement.
    2. Non-verbal cues (blink rate, stress signs).
    3. Final recommendation on their "professional presence".

    Output strictly valid JSON with:
    - engagement_summary
    - stress_analysis
    - professional_presence_score (0-100)
    - interviewer_tip
    """

    try:
        response = await client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {"role": "system", "content": "You are a precise behavioral analyst. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5
        )
        result_json = json.loads(response.choices[0].message.content)
        return {
            "session_id": req.session_id,
            "metrics": metrics,
            "evaluation": result_json,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Behavior LLM Error: {e}")
        raise HTTPException(status_code=500, detail="Behavioral Evaluation Failed")


@app.get("/api/session-summary/{session_id}")
async def session_summary(session_id: str):
    print(f"[GET] /api/session-summary - Session: {session_id}")
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state = sessions[session_id]
    return {
        "session_id": session_id,
        "summary": state.gaze_metrics,
        "total_frames_processed": state.total_frames,
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/analyze/{session_id}")
async def websocket_analysis(ws: WebSocket, session_id: str):
    await ws.accept()
    if session_id not in sessions:
        sessions[session_id] = BehaviorState(deque(maxlen=MAX_BUFFER), deque(maxlen=MAX_BUFFER))
    
    state = sessions[session_id]
    
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            
            if msg.get("type") != "analyze_frame":
                continue
                
            frame = decode_base64_image(msg.get("image_data", ""))
            if frame is None: continue
            
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mesh_results = mp_face_mesh.process(rgb_frame)
            
            face_detected = False
            pitch, yaw = 0.0, 0.0
            
            face_detected = False
            pitch, yaw = 0.0, 0.0
            detection_score = 0.0
            
            if mesh_results.multi_face_landmarks:
                face_detected = True
                lms = mesh_results.multi_face_landmarks[0].landmark
                
                detection_results = mp_face_detection.process(rgb_frame)
                if detection_results.detections:
                    d = detection_results.detections[0]
                    detection_score = d.score[0] # Get confidence score
                    box = d.location_data.relative_bounding_box
                    x, y, fw, fh = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
                    
                    # =========================================================================
                    # Standardize to MediaPipe Head Pose (Switching from ResNet for robustness)
                    # =========================================================================
                    raw_pitch, raw_yaw, _ = get_head_pose_mediapipe(lms, w, h)
                    raw_pitch = max(-60, min(60, raw_pitch))
                    raw_yaw = max(-60, min(60, raw_yaw))
                    # =========================================================================
                    # [OPTION 2]: MediaPipe Head Pose (COMMENTED OUT)
                    # To switch: Comment above lines and Uncomment below lines
                    # =========================================================================
                    # raw_pitch, raw_yaw, _ = get_head_pose_mediapipe(lms, w, h)
                    # raw_pitch = max(-60, min(60, raw_pitch))
                    # raw_yaw = max(-60, min(60, raw_yaw))
                    # =========================================================================
                    
                    # Smoothing
                    alpha = 0.2
                    pitch = alpha * raw_pitch + (1 - alpha) * state.last_pitch
                    yaw = alpha * raw_yaw + (1 - alpha) * state.last_yaw
                    
                    state.last_pitch = pitch
                    state.last_yaw = yaw
                    
                    # Store for stability tracking
                    state.head_pose_history.append((raw_pitch, raw_yaw))
                
                # Blink
                left_ear = calculate_ear(lms, [33, 160, 158, 133, 153, 144])
                right_ear = calculate_ear(lms, [362, 385, 387, 263, 373, 380])
                avg_ear = (left_ear + right_ear) / 2.0
                
                if avg_ear < 0.25:
                    current_time = time.time()
                    if current_time - state.last_blink_time > 0.2: 
                        state.blink_count += 1
                        state.blink_timestamps.append(current_time)
                        state.last_blink_time = current_time
                
                # State Updates
                current_time = time.time()
                state.frame_timestamps.append(current_time)
                # Use balanced 40° threshold for WebSocket
                is_engaged = 1 if (abs(pitch) < 40 and abs(yaw) < 40) else 0
                state.engagement_window.append(is_engaged)
                
                if not is_engaged:
                    print(f"  [DEBUG-WS] Looking Away: Pitch {pitch:.1f}, Yaw {yaw:.1f}")
                
                # Emotion & Stress
                current_emotion, raw_metrics = analyze_emotion_v2(lms)
                ws_eb_raise, ws_m_ratio, ws_m_curve = raw_metrics
                state.emotion_history.append(current_emotion)
                state.gaze_history.append((pitch, yaw)) # Update Gaze History
                state.gaze_metrics["dominant_emotion"] = get_dominant_emotion(state)
                state.gaze_metrics["stress_level"] = analyze_stress_level(state, time.time())
                state.gaze_metrics["confidence"] = calculate_human_confidence(state)
                
                if abs(pitch) < 30 and abs(yaw) < 30: # Use same logic as HTTP
                    state.bg_frames_detected_count += 1
                
                # Rolling window for engagement
                recent_engagement = sum(state.engagement_window)
                total_window = len(state.engagement_window)
                if total_window > 0:
                    state.gaze_metrics["engagement_score"] = float(recent_engagement) / float(total_window)
                else:
                    state.gaze_metrics["engagement_score"] = 0.0
                    
                state.gaze_metrics["blink_rate"] = state.blink_count
            
            state.total_frames += 1
            
            # Terminal Output for WebSocket Frames
            if state.total_frames % 10 == 0:  # Print every 10th frame to avoid spam
                print(f"[WS] Session {session_id} | Frame {state.total_frames} | "
                      f"Pitch: {pitch:+.1f}° | Yaw: {yaw:+.1f}° | "
                      f"Emotion: {state.gaze_metrics['dominant_emotion']} "
                      f"Confidence: {state.gaze_metrics['confidence']*100:.1f}% "
                      f"(E:{ws_eb_raise:.2f} M:{ws_m_ratio:.2f} C:{ws_m_curve:.2f})")
            
            await ws.send_json({
                "type": "analysis_result",
                "session_id": session_id,
                "gaze_analysis": {"pitch": float(pitch), "yaw": float(yaw)} if face_detected else None,
                "behavior_analysis": state.gaze_metrics if face_detected else None,
                "face_detected": face_detected,
                "confidence": state.gaze_metrics["confidence"] if face_detected else None,
                "stress_level": state.gaze_metrics["stress_level"] if face_detected else None,
                "emotion": state.gaze_metrics["dominant_emotion"] if face_detected else None,
                "engagement": state.gaze_metrics["engagement_score"] if face_detected else None,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"WS Error: {e}")

# -----------------------------
# LLM Utilities
# -----------------------------
from openai import AsyncOpenAI

# Role-specific focus areas for smarter prompting
ROLE_FOCUS_AREAS = {
    "Software Engineer": "clean code, system design, time complexity, scalability, corner cases",
    "Frontend Developer": "user experience, accessibility, responsive design, state management, component architecture",
    "Backend Developer": "API design, database optimization, system architecture, security, concurrency",
    "Data Scientist": "statistical validity, data cleaning, feature engineering, model selection, interpretability",
    "Full Stack Developer": "end-to-end understanding, API integration, database design, frontend UX",
    "QA Engineer": "test coverage, edge cases, automation frameworks, bug reporting, regression testing"
}

# Use OpenRouter as seen in .env
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

@app.post("/api/evaluate-answer")
async def evaluate_answer_endpoint(req: AnswerEvaluationRequest):
    print(f"[POST] /api/evaluate-answer - Question: {req.question[:50]}...")
    if not OPENROUTER_API_KEY:
        return {
            "evaluation": {
                "overall_score": 75,
                "detailed_feedback": "Mock evaluation (OpenAI Key missing)."
            }
        }

    # Determine focus area based on role
    focus_area = ROLE_FOCUS_AREAS.get(req.job_role, "core technical competencies and problem-solving skills")

    prompt = f"""
    You are an expert Senior Technical Interviewer at a top-tier tech company. 
    Your goal is to evaluate a candidate's answer for the position of **{req.job_role}**.
    
    **Evaluation Context**:
    - **Identify Key Skills**: Look for evidence of {focus_area}.
    - **Standard**: Expect high-quality, precise, and professional communication.
    
    **Interview Question**: 
    "{req.question}"
    
    **Candidate's Answer**: 
    "{req.answer}"
    
    **Task**:
    Analyze the answer and return a STRICT JSON object with the following fields:
    
    1. **technical_score** (0-100): Accuracy, depth of knowledge, and technical correctness.
    2. **communication_score** (0-100): Clarity, structure, and ability to explain concepts.
    3. **relevance_score** (0-100): How well the answer addresses the specific question asked.
    4. **overall_score** (0-100): Weighted average (Technical 50%, Communication 30%, Relevance 20%).
    5. **strengths** (Array of Strings): 2-3 specific things the candidate did well.
    6. **weaknesses** (Array of Strings): 2-3 specific areas for improvement.
    7. **sentiment** (String): "Positive", "Neutral", or "Negative".
    8. **detailed_feedback** (String): A professional, constructive paragraph (50-80 words) provided directly to the candidate. Start with "You..." and give actionable advice.
    """

    try:
        response = await client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {"role": "system", "content": "You are a strict but fair technical interviewer. Output VALID JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        result_json = json.loads(response.choices[0].message.content)
        return {
            "question": req.question,
            "answer": req.answer,
            "job_role": req.job_role,
            "evaluation": result_json,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"LLM Error: {e}")
        raise HTTPException(status_code=500, detail="AI Evaluation Failed")

        # venv\Scripts\Activate.ps1
        # uvicorn main3:app --reload --port 8002
