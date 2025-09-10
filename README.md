# AI Fitness Coach

A real-time AI-powered fitness coaching application that uses computer vision to track and analyze your workout form. Get instant feedback on your exercises with automatic rep counting and form correction suggestions.

## Features

- Real-time Exercise Detection: Automatically detects Squats and Push-ups
- Form Analysis: AI-powered form correction with instant feedback  
- Automatic Rep Counting: Accurate repetition counting with state machine logic
- Live Web Interface: Beautiful, responsive web UI with real-time video streaming
- Multi-Exercise Support: Extensible architecture for adding more exercises
- Performance Optimized: Smooth 50fps processing with efficient pose estimation

## Supported Exercises

### Current Exercises:
- Squats: Tracks knee angle, depth, and back posture
- Push-ups: Monitors elbow angle, body alignment, and core stability

### Coming Soon:
- Pull-ups
- Lunges  
- Planks
- Bicep Curls

## Prerequisites

- Python 3.10 (Required - MediaPipe does not support Python 3.12+)
- Webcam/Camera
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Installation

### 1. Clone the Repository
```
git clone https://github.com/your-username/ai-fitness-coach.git
cd ai-fitness-coach
```

### 2. Install Python 3.10
Download Python 3.10 from the official website if you don't have it:
https://www.python.org/downloads/release/python-31018/

Verify installation:
```
py --list
```

### 3. Create Virtual Environment
```
py -3.10 -m venv venv
```

### 4. Activate Virtual Environment
On Windows PowerShell:
```
.\venv\Scripts\Activate.ps1
```

On Windows Command Prompt:
```
venv\Scripts\activate.bat
```

On macOS/Linux:
```
source venv/bin/activate
```

### 5. Install Dependencies
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run the Application
```
python app.py
```

### 7. Open Your Browser
Navigate to http://localhost:8002 to start your AI fitness coaching session

## Configuration

You can modify the following settings in app.py:

Camera Configuration:
```
CAM_INDEX = 0  # Change if you have multiple cameras
```

Performance Settings:
```
JPEG_QUALITY = 70  # Adjust for quality vs bandwidth
LOOP_THROTTLE_SECONDS = 0.02  # ~50fps processing rate
```

Exercise Thresholds (in RepCounter classes):
```
down_threshold = 100  # Angle threshold for "down" position
up_threshold = 160    # Angle threshold for "up" position
```

## Usage

1. Start the Application: Click the "Start" button in the web interface
2. Position Yourself: Ensure your full body is visible in the camera frame
3. Begin Exercising: Start performing squats or push-ups
4. Get Real-time Feedback:
   - Rep counts update automatically
   - Form issues are highlighted in the feedback panel
   - Exercise status shows your current movement phase

### Tips for Best Results:
- Ensure good lighting
- Wear contrasting colors to your background
- Keep your full body in frame
- Stand 3-6 feet from the camera

## Architecture

### Core Components:

- FastAPI Backend: Handles WebSocket connections and video processing
- MediaPipe: Google's pose estimation for landmark detection
- OpenCV: Computer vision and image processing
- Exercise Classes: Modular exercise detection and tracking system
- Rep Counter: State machine for accurate repetition counting

### Key Files:
```
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Adding New Exercises

The application is designed for easy extensibility. To add a new exercise:

1. Create a new class inheriting from Exercise:
```python
class NewExercise(Exercise):
    def __init__(self):
        super().__init__()
        self.rep_counter = RepCounter(down_threshold=X, up_threshold=Y)
        self.name = "NewExercise"
    
    def detect(self, landmarks):
        # Implement detection logic
        return True/False
    
    def track(self, landmarks):
        # Implement tracking and form analysis
        return count, feedback, form_issues
```

2. Add to the exercise registry:
```python
EXERCISE_REGISTRY.append(NewExercise())
```

## Troubleshooting

### Common Issues:

**Camera not working**:
- Check if your camera is being used by another application
- Try changing CAM_INDEX to 1, 2, etc.
- Verify camera permissions

**Poor detection accuracy**:
- Improve lighting conditions
- Ensure full body visibility
- Check camera angle and distance

**Performance issues**:
- Reduce JPEG_QUALITY for better performance
- Increase LOOP_THROTTLE_SECONDS for lower CPU usage
- Close other resource-intensive applications

**WebSocket connection fails**:
- Check if port 8002 is available
- Try running on a different port by modifying the uvicorn.run() call

**MediaPipe installation error**:
- Ensure you're using Python 3.10 (not 3.11+ or 3.12+)
- Recreate your virtual environment with py -3.10 -m venv venv

## Technical Details

- Pose Estimation: MediaPipe Pose with 33 body landmarks
- Angle Calculation: 3D vector mathematics for joint angles
- State Machine: Robust rep counting with configurable thresholds
- Real-time Processing: WebSocket streaming at ~50fps
- Cross-platform: Works on Windows, macOS, and Linux

## Tech Stack

- Backend: FastAPI, WebSockets
- Computer Vision: OpenCV, MediaPipe Pose
- Numerical Computing: NumPy  
- Frontend: HTML, JavaScript (WebSocket client)
- Language: Python 3.10

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-exercise)
3. Commit your changes (git commit -m 'Add amazing new exercise')
4. Push to the branch (git push origin feature/amazing-exercise)
5. Open a Pull Request

### Areas for Contribution:
- New exercise implementations
- UI/UX improvements
- Performance optimizations
- Mobile responsiveness
- Exercise history tracking
- Workout routines

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe by Google for pose estimation
- FastAPI for the web framework  
- OpenCV for computer vision capabilities

## Support

If you encounter any issues or have questions:

1. Check the Issues page on GitHub
2. Create a new issue with detailed information
3. Include your Python version, OS, and error messages