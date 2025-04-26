🍛 Smart Food Pricing System
--------------------------------

Welcome to the Smart Food Pricing System!
This is a smart, AI-powered food pricing system that brings transparency, automation, and efficiency to everyday dining. In busy food courts or canteens, pricing mixed rice or buffet items is often manual and inconsistent. Our system uses AI to instantly recognize food items on a plate and calculate the price, improving user experience and saving operational time — a step toward smarter, tech-enabled urban life

--------------------------------

## 📥 Getting Started

### 1) Clone the repo or download the project files
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
exit for bash
```

### 2) Install the required libraries (preferably in a virtual environment)
```bash
pip install streamlit ultralytics pandas pillow numpy
```

### 3) Modify the path to your trained YOLO model inside the Python file
In your Python script, set the model path:
```bash

from ultralytics import YOLO

model = YOLO(r"path\to\your\best.pt")
```

### 4) Launch the app with:
```bash

streamlit run app.py
```

Then upload a food image and instantly view the pricing estimation!

--------------------------------

🛠️ Tech Stack
----------------------------------
Streamlit - For building the web interface

Ultralytics YOLO - For object detection

Pandas, Pillow, NumPy - For data handling and image processing

Python - Core programming language

--------------------------------

🖼️ Example Usage
---------------------
1. Upload your meal image.
2. The app detects food items and measures the portion size.
3. Instantly view the estimated price based on detected portions.

--------------------------------

