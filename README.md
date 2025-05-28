# Coconut Detector using YOLOv8

## Overview
This project implements a Coconut Detector powered by **YOLOv8** — an advanced real-time object detection model. It detects and counts fresh tender coconuts from both **image** and **video** inputs. The system automatically generates a PDF report summarizing the total coconut count along with detailed frame-wise statistics.

The solution aims to streamline agricultural monitoring processes by providing accurate and automated coconut detection, reducing manual labor, and improving harvesting efficiency.

---

## Features
- Detects fresh tender coconuts in images and videos using YOLOv8.
- Supports both image and video input formats.
- Generates an automated PDF report with coconut counts and frame-wise data.
- Easy-to-use interface for running detection and generating reports.
- Utilizes Python, OpenCV, and YOLOv8 for fast and accurate detection.

---

## Getting Started

### Prerequisites
- Python 3.7+
- Git
- Required Python libraries (listed in `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Avanthika-06/Coconut_Detector.git
   cd Coconut_Detector

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
    ```bash
   pip install -r requirements.txt

## Usage
Run Detection on Image or Video-- 
   python detect.py --source path/to/image_or_video

## Getting Started
After detection, a PDF report summarizing the total coconut count and frame-wise details will be automatically generated and saved in the output directory.

## Project Structure

├── detect.py          # Main detection script
├── model             # YOLOv8 model weights
├── outputs           # Directory where output videos, images, and PDF reports are saved
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
└── utils             # Utility scripts (optional)

### License
This project is licensed under the MIT License. See the LICENSE file for details.

    
