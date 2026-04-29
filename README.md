# Robot Automation and Vision System (ELEC3-NERV)

This project is an autonomous robot system designed to find sponges using computer vision, navigate toward them, and pick them up with a robotic arm. Once a sponge is secured, the robot delivers it to a designated ArUco marker (ID 1).

## Overview of Project Files

### Python Scripts

The vision and high-level decision-making are handled by these main scripts:

- **auto.py**: This is the core automation script. It runs the main logic loop, processes camera frames for sponge and marker detection, and sends movement commands to the Arduino.
- **detect_arucos.py**: Use this script for testing ArUco marker detection. It helps verify distance estimation and can be used to manually trigger the dropping sequence for testing.
- **calibrate_camera.py**: A specialized tool to calibrate your webcam using a single ArUco marker. It calculates the camera matrix and distortion coefficients, saving them to a `.npz` file to enable accurate real-world distance estimation in other scripts.
- **sponge_training.py**: The training node for the YOLOv8 model. It handles dataset extraction from a zip file, performs data augmentation, trains the model, and exports the final weights.

### Arduino Firmware

The low-level hardware control is located in **robot/body/body.ino**. This sketch handles:

- Driving the DC motors via an L298N driver.
- Coordinating the 7-DOF robotic arm (Base, Shoulder, Elbow, Wrist, and Gripper).
- Monitoring an HC-SR04 ultrasonic sensor for basic front-facing obstacle detection.

### Vision Models

The system uses YOLOv8 for object detection:

- **model/best.pt**: The standard PyTorch model weights.
- **model/best.onnx**: An optimized ONNX version of the model, which typically provides better performance on edge devices like the Raspberry Pi.

## Getting Started

### Python Environment Setup

You will need Python 3.8 or higher. To install the necessary libraries, run:

```bash
pip install -r requirements.txt
```

This will install `ultralytics`, `opencv-contrib-python`, `pyserial`, `numpy`, `torch`, and `pyyaml`.

### Arduino Configuration

1. Open the **robot/body/body.ino** file in the Arduino IDE.
2. The code uses the standard **Servo.h** library.
3. Upload the sketch to your board (e.g., Arduino Mega or Uno) on the appropriate COM port.

## Running the System

### Full Automation

To start the robot's autonomous search and delivery sequence, run:

```bash
python auto.py
```

The script will attempt to find the Arduino automatically and start the search routine.

### Testing Markers

To verify the marker detection and pose estimation setup:

```bash
python detect_arucos.py --family DICT_APRILTAG_36H11 --marker-size 0.2
```

### Model Training

To train a new YOLO model using your own dataset:

1. Place your dataset zip (e.g., `ELEC3.yolov8.zip`) in the root directory.
2. Optional: Add background images to a `backgrounds_for_yolo` folder for hard negative training.
3. Run the training script:

```bash
python sponge_training.py
```

This will extract the data, train the model for 100 epochs (by default), and export the `best.pt` file.

### Camera Calibration

If the distance readings in `detect_arucos.py` or the robot's navigation seem inaccurate, you should perform a manual camera calibration. 

1. Run the calibration script in webcam mode:
```bash
python calibrate_camera.py --mode webcam --captures 40 --family DICT_APRILTAG_36H11
```
2. Hold an ArUco marker (ID 0) in front of the camera and move it to different positions, angles, and distances. Press 'c' to capture a frame. 
3. Once 40 frames are captured, it will save `camera_calibration.npz`.
4. You can then use this file with the detection script:
```bash
python detect_arucos.py --calibration camera_calibration.npz
```

## Hardware Pinout

For quick reference, here are the main pin assignments on the Arduino:

- **Motors**: IN1-IN4 (Pins 22, 24, 26, 28)
- **Servos**: Pins 6 through 12
- **Ultrasonic Sensor**: Trig (Pin 30), Echo (Pin 31)
- **Baud Rate**: 115200
