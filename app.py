import asyncio
import base64
import json
from typing import Dict, List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# --- Configuration ---
CAM_INDEX = 0
JPEG_QUALITY = 70
LOOP_THROTTLE_SECONDS = 0.02  # Approx 50fps, but processing time will dominate

app = FastAPI()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Utility Functions ---

def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Calculates the angle at point b in degrees."""
    a_np, b_np, c_np = np.array(a), np.array(b), np.array(c)
    ba, bc = a_np - b_np, c_np - b_np
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos_angle = np.dot(ba, bc) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def landmarks_to_dict(landmarks, image_width: int, image_height: int) -> Dict[str, Tuple[float, float]]:
    """Converts MediaPipe landmarks to a dictionary of normalized coordinates."""
    # The correct way to get the landmark names is from the enum itself
    landmark_names = {lm.value: lm.name for lm in mp_pose.PoseLandmark}
    
    # Iterate over the detected landmarks from the results object
    return {
        landmark_names[i]: (lm.x, lm.y)
        for i, lm in enumerate(landmarks)
    }

# --- Exercise Logic Abstraction ---

class RepCounter:
    """A state machine for counting repetitions of an exercise."""
    def __init__(self, down_threshold: float, up_threshold: float):
        self.count = 0
        self.in_down_phase = False
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold

    def update(self, angle: float) -> Tuple[int, str]:
        """Updates the counter based on the current angle."""
        feedback = "Moving"
        if angle < self.down_threshold:
            if not self.in_down_phase:
                self.in_down_phase = True
                feedback = "Down"
        elif angle > self.up_threshold:
            if self.in_down_phase:
                self.count += 1
                self.in_down_phase = False
                feedback = "Rep Counted!"
            else:
                feedback = "Up"
        return self.count, feedback

class Exercise:
    """Base class for an exercise, defining the interface."""
    def __init__(self):
        self.rep_counter: Optional[RepCounter] = None

    def detect(self, landmarks: Dict[str, Tuple[float, float]]) -> bool:
        """Heuristic to detect if this exercise is being performed."""
        raise NotImplementedError

    def track(self, landmarks: Dict[str, Tuple[float, float]]) -> Tuple[int, str, List[str]]:
        """Tracks reps and form for this exercise."""
        raise NotImplementedError
        
    def get_count(self) -> int:
        return self.rep_counter.count if self.rep_counter else 0

class Squat(Exercise):
    def __init__(self):
        super().__init__()
        self.rep_counter = RepCounter(down_threshold=100, up_threshold=160)
        self.name = "Squat"

    def detect(self, landmarks: Dict[str, Tuple[float, float]]) -> bool:
        # Heuristic: User is upright. Shoulders y < hips y
        if "LEFT_SHOULDER" not in landmarks or "LEFT_HIP" not in landmarks:
            return False
        return landmarks["LEFT_SHOULDER"][1] < landmarks["LEFT_HIP"][1]

    def track(self, landmarks: Dict[str, Tuple[float, float]]) -> Tuple[int, str, List[str]]:
        # Check for required landmarks
        required = ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "LEFT_SHOULDER"]
        if not all(k in landmarks for k in required):
            return self.get_count(), "Make sure your full body is visible.", []

        # Calculate angles
        left_knee_angle = calculate_angle(landmarks["LEFT_HIP"], landmarks["LEFT_KNEE"], landmarks["LEFT_ANKLE"])
        right_knee_angle = calculate_angle(landmarks["RIGHT_HIP"], landmarks["RIGHT_KNEE"], landmarks["RIGHT_ANKLE"])
        knee_angle = (left_knee_angle + right_knee_angle) / 2.0
        
        # Update rep counter
        count, rep_feedback = self.rep_counter.update(knee_angle)

        # Form checks
        form_issues = []
        if knee_angle > 90 and self.rep_counter.in_down_phase:
            form_issues.append("Go deeper to reach parallel.")
        
        left_back_angle = calculate_angle(landmarks["LEFT_SHOULDER"], landmarks["LEFT_HIP"], landmarks["LEFT_KNEE"])
        if left_back_angle < 70:
             form_issues.append("Keep your chest up and back straight.")

        return count, rep_feedback, form_issues

class Pushup(Exercise):
    def __init__(self):
        super().__init__()
        self.rep_counter = RepCounter(down_threshold=80, up_threshold=160)
        self.name = "Pushup"

    def detect(self, landmarks: Dict[str, Tuple[float, float]]) -> bool:
        # Heuristic: Shoulders and hips are roughly horizontal
        required = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
        if not all(k in landmarks for k in required):
            return False
        
        shoulders_y = (landmarks["LEFT_SHOULDER"][1] + landmarks["RIGHT_SHOULDER"][1]) / 2
        hips_y = (landmarks["LEFT_HIP"][1] + landmarks["RIGHT_HIP"][1]) / 2
        return abs(shoulders_y - hips_y) < 0.1

    def track(self, landmarks: Dict[str, Tuple[float, float]]) -> Tuple[int, str, List[str]]:
        required = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "LEFT_HIP", "LEFT_ANKLE"]
        if not all(k in landmarks for k in required):
            return self.get_count(), "Make sure your full body is visible.", []

        # Calculate angles
        left_elbow_angle = calculate_angle(landmarks["LEFT_SHOULDER"], landmarks["LEFT_ELBOW"], landmarks["LEFT_WRIST"])
        right_elbow_angle = calculate_angle(landmarks["RIGHT_SHOULDER"], landmarks["RIGHT_ELBOW"], landmarks["RIGHT_WRIST"])
        elbow_angle = (left_elbow_angle + right_elbow_angle) / 2.0
        
        count, rep_feedback = self.rep_counter.update(elbow_angle)

        # Form checks
        form_issues = []
        shoulder_y = landmarks["LEFT_SHOULDER"][1]
        hip_y = landmarks["LEFT_HIP"][1]
        ankle_y = landmarks["LEFT_ANKLE"][1]

        # Check if hips are sagging or piking
        if not (shoulder_y < hip_y < ankle_y or shoulder_y > hip_y > ankle_y):
             if hip_y > max(shoulder_y, ankle_y) + 0.05:
                form_issues.append("Hips sagging. Engage your core.")
             if hip_y < min(shoulder_y, ankle_y) - 0.05:
                form_issues.append("Hips too high. Lower them.")
                
        return count, rep_feedback, form_issues

# --- Exercise Registry ---
# This makes it easy to add more exercises in the future!
EXERCISE_REGISTRY: List[Exercise] = [Squat(), Pushup()]


# --- Frame Annotation ---

def annotate_frame(
    frame: np.ndarray, 
    landmarks, 
    counters: Dict[str, int], 
    exercise_name: str, 
    feedback: str, 
    form_issues: List[str]
) -> np.ndarray:
    """Draws all the coaching information on the frame."""
    annotated = frame.copy()
    h, w, _ = annotated.shape

    # Draw landmarks
    if landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    # --- UI Overlay ---
    # Top Status Bar
    cv2.rectangle(annotated, (0, 0), (w, 80), (25, 25, 25), -1)
    cv2.putText(annotated, "AI FIT COACH", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Exercise and Feedback
    cv2.putText(annotated, f"EXERCISE: {exercise_name.upper()}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 191, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"STATUS: {feedback}", (w - 400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Rep Counters
    counter_x_pos = w - 180
    cv2.putText(annotated, "REPS", (counter_x_pos, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for i, (ex, count) in enumerate(counters.items()):
        y_pos = 50 + i * 20
        cv2.putText(annotated, f"{ex.title()}: {count}", (counter_x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Form Issues Box
    if form_issues:
        y_start = h - 20 - (len(form_issues) * 30)
        cv2.rectangle(annotated, (10, y_start - 10), (w - 10, h - 10), (0, 0, 128), -1)
        cv2.rectangle(annotated, (10, y_start - 10), (w - 10, h - 10), (0, 0, 255), 2)
        for i, issue in enumerate(form_issues):
            cv2.putText(annotated, issue, (25, y_start + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


# --- FastAPI Endpoints ---

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Fitness Coach ✨</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #1a1a2e;
            --primary-color: #16213e;
            --secondary-color: #0f3460;
            --accent-color: #e94560;
            --text-color: #dcdcdc;
            --glow-color: #39ff14;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Poppins', sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            display: flex;
            justify-content: center;
            align-items: flex-start;
            gap: 30px;
            padding: 30px;
            min-height: 100vh;
            opacity: 0;
            animation: fadeIn 0.5s ease-out forwards;
        }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        
        .container {
            display: flex;
            gap: 30px;
            width: 100%;
            max-width: 1400px;
        }
        .main-content { flex: 3; }
        .sidebar { flex: 1; display: flex; flex-direction: column; gap: 20px; }

        .card {
            background: rgba(22, 33, 62, 0.7);
            border: 1px solid var(--secondary-color);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: all 0.3s ease;
        }
        .card:hover { transform: translateY(-5px); box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45); }
        
        .video-box {
            position: relative;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
        }
        #video {
            display: block;
            width: 100%;
            height: auto;
            transform: scaleX(-1); /* Selfie view */
        }

        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .header h1 {
            font-size: 24px;
            color: white;
            font-weight: 600;
        }
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            padding: 6px 12px;
            border-radius: 20px;
            background: rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; }
        .status.disconnected .status-dot { background-color: #e94560; }
        .status.connecting .status-dot { background-color: #fca120; animation: pulse-orange 1.5s infinite; }
        .status.live .status-dot { background-color: var(--glow-color); box-shadow: 0 0 10px var(--glow-color); }
        
        .controls button {
            padding: 12px 20px;
            border-radius: 8px;
            background: var(--accent-color);
            color: white;
            border: none;
            cursor: pointer;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            transition: all 0.3s ease;
            margin-left: 10px;
        }
        .controls button:hover:not(:disabled) { background: #ff6384; transform: scale(1.05); }
        .controls button:disabled { background: #555; cursor: not-allowed; }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-item .label {
            font-size: 14px;
            color: #a7a7a7;
            margin-bottom: 5px;
        }
        .stat-item .value {
            font-size: 48px;
            font-weight: 700;
            color: white;
            line-height: 1;
            transition: color 0.3s ease, transform 0.2s ease;
        }
        .pulse-animation {
            animation: pulse-green 0.5s ease-out;
        }
        
        .feedback-box .label {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        #exercise { color: var(--accent-color); font-weight: 700; }
        #feedback { font-size: 14px; min-height: 20px; font-style: italic; }
        #form-issues { margin-top: 15px; padding-left: 20px; list-style-type: '⚠️ '; }
        #form-issues li { margin-bottom: 8px; font-size: 13px; color: #ffc107; }

        @keyframes pulse-green {
            0% { transform: scale(1); color: var(--glow-color); text-shadow: 0 0 15px var(--glow-color); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); color: white; text-shadow: none; }
        }
        @keyframes pulse-orange {
            0% { box-shadow: 0 0 0 0 rgba(252, 161, 32, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(252, 161, 32, 0); }
            100% { box-shadow: 0 0 0 0 rgba(252, 161, 32, 0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-content">
            <div class="card video-box">
                <img id="video" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" alt="AI Fitness Coach Stream"/>
            </div>
        </div>
        <div class="sidebar">
            <div class="card header">
                <div>
                    <h1>AI Coach</h1>
                    <div id="status" class="status disconnected">
                        <div class="status-dot"></div>
                        <span id="status-text">Disconnected</span>
                    </div>
                </div>
                <div class="controls">
                    <button id="startBtn">Start</button>
                    <button id="stopBtn" disabled>Stop</button>
                </div>
            </div>
            <div class="card">
                <div class="feedback-box">
                    <div class="label">Current Exercise: <span id="exercise">—</span></div>
                    <div id="feedback">Waiting to start...</div>
                </div>
            </div>
            <div class="card">
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Squats</div>
                        <div id="squats" class="value">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Pushups</div>
                        <div id="pushups" class="value">0</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="feedback-box">
                    <div class="label">Form Analysis</div>
                    <ul id="form-issues">
                        <li>No issues detected yet.</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

<script>
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusDiv = document.getElementById('status');
    const statusText = document.getElementById('status-text');
    const videoImg = document.getElementById('video');
    
    const exerciseEl = document.getElementById('exercise');
    const feedbackEl = document.getElementById('feedback');
    const formIssuesUl = document.getElementById('form-issues');
    
    const counters = {
        squat: document.getElementById('squats'),
        pushup: document.getElementById('pushups')
    };
    
    let ws = null;

    function setStatus(state, text) {
        statusDiv.className = 'status ' + state;
        statusText.textContent = text;
    }

    function connect() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log("Already connected.");
            return;
        }

        setStatus('connecting', 'Connecting...');
        startBtn.disabled = true;

        const protocol = window.location.protocol === "https:" ? "wss" : "ws";
        ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

        ws.onopen = () => {
            console.log("WebSocket connection established.");
            setStatus('live', 'Live');
            stopBtn.disabled = false;
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.error) {
                    console.error("Backend Error:", data.error);
                    alert(`Error from server: ${data.error}`);
                    disconnect();
                    return;
                }

                videoImg.src = "data:image/jpeg;base64," + data.frame;
                
                exerciseEl.textContent = data.exercise === 'unknown' ? '—' : data.exercise.toUpperCase();
                feedbackEl.textContent = data.feedback;

                // Update counters with animation
                for (const [exercise, el] of Object.entries(counters)) {
                    const newCount = data.counters[exercise] || 0;
                    if (el.textContent !== newCount.toString()) {
                        el.textContent = newCount;
                        el.classList.add('pulse-animation');
                        setTimeout(() => el.classList.remove('pulse-animation'), 500);
                    }
                }
                
                // Update form issues
                formIssuesUl.innerHTML = '';
                if (data.form_issues && data.form_issues.length > 0) {
                    data.form_issues.forEach(issue => {
                        const li = document.createElement('li');
                        li.textContent = issue;
                        formIssuesUl.appendChild(li);
                    });
                } else {
                    const li = document.createElement('li');
                    li.textContent = 'Great form!';
                    li.style.color = '#39ff14'; // Green for good form
                    formIssuesUl.appendChild(li);
                }

            } catch (e) {
                console.error("Failed to parse message or update UI:", e);
            }
        };

        ws.onclose = () => {
            console.log("WebSocket connection closed.");
            setStatus('disconnected', 'Disconnected');
            startBtn.disabled = false;
            stopBtn.disabled = true;
            ws = null;
        };

        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            setStatus('disconnected', 'Error');
            alert("Could not connect to the server. Please ensure it is running.");
            startBtn.disabled = false;
            stopBtn.disabled = true;
        };
    }

    function disconnect() {
        if (ws) {
            ws.close();
        }
    }

    startBtn.onclick = connect;
    stopBtn.onclick = disconnect;

</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        error_payload = json.dumps({"error": f"Could not open camera with index {CAM_INDEX}."})
        await websocket.send_text(error_payload)
        await websocket.close()
        return

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            current_exercise: Optional[Exercise] = None

            while True:
                ret, frame = await asyncio.get_event_loop().run_in_executor(None, cap.read)
                if not ret:
                    await asyncio.sleep(0.1)
                    continue

                # --- IMPORTANT CHANGE STARTS HERE ---
                # 1. Do NOT flip the frame yet. Process the original frame.
                original_frame = frame.copy() # Keep a copy if you need the original later, though not strictly necessary here
                h, w, _ = original_frame.shape
                rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

                results = await asyncio.get_event_loop().run_in_executor(None, pose.process, rgb_frame)

                exercise_name, feedback, form_issues = "unknown", "No person detected", []

                if results.pose_landmarks:
                    landmarks_dict = landmarks_to_dict(results.pose_landmarks.landmark, w, h) # Renamed to avoid conflict with `landmarks` in the next line
                    
                    # Detect or stay with current exercise
                    if current_exercise is None or not current_exercise.detect(landmarks_dict):
                        current_exercise = next((ex for ex in EXERCISE_REGISTRY if ex.detect(landmarks_dict)), None)

                    if current_exercise:
                        exercise_name = current_exercise.name
                        _, feedback, form_issues = current_exercise.track(landmarks_dict)
                    else:
                        feedback = "Stand in position for Squat or Pushup."
                else:
                    current_exercise = None # Reset if no person is detected

                counters = {ex.name.lower(): ex.get_count() for ex in EXERCISE_REGISTRY}

                # 2. Annotate the ORIGINAL frame
                annotated_frame = annotate_frame(original_frame, results.pose_landmarks, counters, exercise_name, feedback, form_issues)
                
                # 3. NOW, flip the fully annotated frame for selfie view
                final_display_frame = cv2.flip(annotated_frame, 1)
                # --- IMPORTANT CHANGE ENDS HERE ---

                ret, jpeg = cv2.imencode('.jpg', final_display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if not ret:
                    continue

                b64_frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')

                payload = {
                    "frame": b64_frame,
                    "exercise": exercise_name.lower(),
                    "feedback": feedback,
                    "form_issues": form_issues,
                    "counters": counters
                }

                await websocket.send_text(json.dumps(payload))
                await asyncio.sleep(LOOP_THROTTLE_SECONDS)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        try:
            await websocket.send_text(json.dumps({"error": "An unexpected server error occurred."}))
        except:
            pass # Client might already be gone
    finally:
        print("Closing resources.")
        cap.release()
        try:
            await websocket.close()
        except:
            pass
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8002, 
        reload=True
    )