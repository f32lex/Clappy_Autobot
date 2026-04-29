#!/usr/bin/env python3
import cv2
import serial
import serial.tools.list_ports
import time
import os
from ultralytics import YOLO

# CONFIGURATION
PT_MODEL_PATH = "/home/markmarkmark/elec3-nerv/model/best.pt"
ONNX_MODEL_PATH = "/home/markmarkmark/elec3-nerv/model/best.onnx"
CONFIDENCE_THRESHOLD = 0.5  
CAMERA_INDEX = 0  

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(x in p.description or x in p.device for x in ["Arduino", "ACM", "USB", "ttyACM", "ttyUSB"]):
            return p.device
    return None

def main():
    # 1. Load the model
    if not os.path.exists(ONNX_MODEL_PATH):
        print("ONNX model not found. Exporting from .pt for better speed...")
        temp_model = YOLO(PT_MODEL_PATH)
        temp_model.export(format="onnx", imgsz=320)
    
    print("Loading optimized ONNX model...")
    model = YOLO(ONNX_MODEL_PATH, task='detect')
    print("Model loaded successfully!")
    
    # 2. Initialize Serial
    port = find_arduino_port()
    if not port:
        print("!!! No Arduino found.")
        return
    
    try:
        ser = serial.Serial(port, 115200, timeout=0.05)
    except serial.SerialException as e:
        print(f"!!! Could not open serial port {port}.")
        print(f"Error: {e}")
        return

    time.sleep(3) 
    ser.reset_input_buffer()
    print(f"Connected to Arduino on {port}")

    # 3. Initialize Camera 
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("!!! Could not open webcam. Ensure no other apps are using it.")
        ser.close()
        return

    print("Robot Online. Starting Search...")
    
    # --- STATE DEFINITIONS ---
    STATE_SEARCHING = "SEARCHING"
    STATE_APPROACHING_SPONGE = "APPROACHING_SPONGE"
    STATE_PICKING = "PICKING"
    STATE_DELIVERING = "DELIVERING"
    STATE_APPROACHING_MARKER = "APPROACHING_MARKER"
    STATE_DROPPING = "DROPPING"

    current_state = STATE_SEARCHING
    ser.write(b'F') # Start moving
    
    # Center calibration
    IMG_CENTER_X = 320
    STEER_THRESHOLD = 50 # Pixels off center to trigger a turn
    PICKUP_HEIGHT_THRESHOLD = 300 # Sponge box height when close enough to pick
    DROP_MARKER_SIZE_THRESHOLD = 150 # Marker width/height when close enough to drop

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # --- YOLO DETECTION ---
            results = list(model(frame, conf=CONFIDENCE_THRESHOLD, stream=True, imgsz=320))
            
            target_sponge = None
            if results:
                for r in results:
                    for box in r.boxes:
                        # Get the largest/closest sponge if multiple
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = (x1 + x2) / 2
                        h = y2 - y1
                        target_sponge = {'cx': cx, 'h': h, 'box': (int(x1), int(y1), int(x2), int(y2))}
                        break # Just take the first one for now

            # --- ARUCO DETECTION ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
            try:
                parameters = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, _ = detector.detectMarkers(gray)
            except AttributeError:
                parameters = cv2.aruco.DetectorParameters_create()
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            target_marker = None
            if ids is not None:
                for i, marker_id in enumerate(ids):
                    if marker_id[0] == 1:
                        c = corners[i][0]
                        cx = (c[0][0] + c[2][0]) / 2 # Center X of marker
                        w = abs(c[0][0] - c[2][0])   # Width of marker
                        target_marker = {'cx': cx, 'w': w, 'corners': corners[i]}
                        break

            # --- STATE MACHINE LOGIC ---
            
            # Read Serial for "DONE" signal or Obstacles
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"Arduino: {line}")
                if "DONE" in line:
                    if current_state == STATE_PICKING:
                        print(">>> Pickup Complete. Now hunting for ID 1...")
                        current_state = STATE_DELIVERING
                        ser.write(b'F')
                    elif current_state == STATE_DROPPING:
                        print(">>> Drop Complete. Returning to Search...")
                        current_state = STATE_SEARCHING
                        ser.write(b'F')

            # State Actions
            if current_state == STATE_SEARCHING:
                if target_sponge:
                    print(">>> Sponge Sighted! Approaching...")
                    current_state = STATE_APPROACHING_SPONGE
                else:
                    ser.write(b'F') # Keep cruising

            elif current_state == STATE_APPROACHING_SPONGE:
                if not target_sponge:
                    print(">>> Lost Sponge. Searching again...")
                    current_state = STATE_SEARCHING
                else:
                    # Steering Logic
                    error = target_sponge['cx'] - IMG_CENTER_X
                    if target_sponge['h'] > PICKUP_HEIGHT_THRESHOLD:
                        print(">>> Target in Range! Triggering Pickup ('A')")
                        ser.write(b'A')
                        current_state = STATE_PICKING
                    elif error > STEER_THRESHOLD:
                        ser.write(b'R')
                    elif error < -STEER_THRESHOLD:
                        ser.write(b'L')
                    else:
                        ser.write(b'F')

            elif current_state == STATE_DELIVERING:
                if target_marker:
                    print(">>> ArUco ID 1 Sighted! Approaching...")
                    current_state = STATE_APPROACHING_MARKER
                else:
                    ser.write(b'F') # Keep cruising looking for marker

            elif current_state == STATE_APPROACHING_MARKER:
                if not target_marker:
                    print(">>> Lost Marker. Returning to Deliver mode...")
                    current_state = STATE_DELIVERING
                else:
                    # Steering Logic
                    error = target_marker['cx'] - IMG_CENTER_X
                    if target_marker['w'] > DROP_MARKER_SIZE_THRESHOLD:
                        print(">>> Drop Zone Reached! Triggering Drop ('D')")
                        ser.write(b'D')
                        current_state = STATE_DROPPING
                    elif error > STEER_THRESHOLD:
                        ser.write(b'R')
                    elif error < -STEER_THRESHOLD:
                        ser.write(b'L')
                    else:
                        ser.write(b'F')

            # Visual Debugging (Optional)
            if target_sponge:
                x1, y1, x2, y2 = target_sponge['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if target_marker:
                cv2.aruco.drawDetectedMarkers(frame, [target_marker['corners']], None)

            # cv2.imshow("Robot Vision", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        ser.write(b'S')
        ser.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
