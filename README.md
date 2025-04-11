#  Pose Angle Analyser

This project analyzes human joint angles from videos using **MediaPipe Pose** and **OpenCV**. Each Python file in the repository corresponds to a specific **body part** (e.g., `neck.py`, `legs.py`, `knee.py`). When run, the script:

- Captures or loads a video of that joint in motion.
- Detects relevant pose landmarks using MediaPipe.
- Calculates the angle at that joint.
- Displays the live or recorded video with the measured angle overlaid.

---


Each script is focused on analyzing a different body part.

---

## 🛠 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/nimitsheth/pose-angle-analyzer.git
cd pose-angle-analyser
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate 
```

3. **Install the dependencies**

```bash
pip install -r requirements.txt 
```

## ▶️ Running the Code

To run the analyzer for a specific joint, just execute the corresponding file. For example, to analyze the knee:

```bash
python knee.py
```
The program will or load a preexisting video, detect the pose, and display the angle of the joint in real-time.

## 🧠 Technologies Used

- MediaPipe Pose – for pose estimation.

- OpenCV – for video capture and drawing.

- Python – core scripting language.