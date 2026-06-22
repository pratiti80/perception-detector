# perception-detector
---
## Overview

This perception detector is a computer vision pipeline that utilizes YOLOv8 to detect cars 
and trucks and calculate their distance from the camera in real time.
## Demo


## Features

- Vehicle Detection
- Multi-Vehicle Tracking
- Vehicle Distance Calculation
- Time to Collision calculation
- Color Coded Time to Collision Alerts

## How It Works

This perception detector uses YOLOv8 to detect cars and trucks and BotSort to assign them a persistent tracking ID. It then uses the pixel width and estimated real width, 
along with the focal length of the camera, to calculate the distance of the objects from the camera. The time to collision is calculated by dividing the current distance 
of the car by the relative speed of it. TTC values are color-coded on screen: green for safe, yellow for caution, and red for imminent collision.

## Tech Stack

- **Python**
- **YOLOv8** (Ultralytics) — object detection
- **OpenCV** — video I/O and visualization
- **BotSort** — multi-object tracking

## Installation

```bash
git clone https://github.com/pratiti80/perception-detector.git
cd perception-detector
pip install -r requirements.txt
```

## Usage

```bash
python distance_estimator.py --video path/to/your/video.mp4
```
