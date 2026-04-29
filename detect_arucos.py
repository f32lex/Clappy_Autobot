import argparse
import os
import cv2
import numpy as np
import cv2.aruco as aruco
import serial
import time

FAMILIES = {
    "DICT_6X6_250":         aruco.DICT_6X6_250,
    "DICT_6X6_1000":        aruco.DICT_6X6_1000,
    "DICT_5X5_1000":        aruco.DICT_5X5_1000,
    "DICT_7X7_1000":        aruco.DICT_7X7_1000,
    "DICT_APRILTAG_36H11":  aruco.DICT_APRILTAG_36H11,
    "DICT_6X6_50":          aruco.DICT_6X6_50,
    "DICT_6X6_100":         aruco.DICT_6X6_100,
}

def get_aruco_dict(family_name):
    if family_name not in FAMILIES:
        raise ValueError(f"Unknown family '{family_name}'. Choose from: {', '.join(FAMILIES.keys())}")
    
    try:
        return aruco.getPredefinedDictionary(FAMILIES[family_name])
    except AttributeError:
        return aruco.Dictionary_get(FAMILIES[family_name])

def set_camera_resolution(cap, resolution_str):
    if resolution_str:
        try:
            w, h = map(int, resolution_str.lower().split('x'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        except Exception as e:
            print(f"Warning: Could not set camera resolution. Error: {e}")

def get_dummy_camera_matrix(image_shape):
    """
    Create a basic camera matrix assuming a typical webcam field of view (~60 degrees).
    This provides an ESTIMATION. For accurate distance, proper camera calibration is needed.
    """
    h, w = image_shape[:2]
    # Rough approximation of focal length in pixels
    focal_length = w
    center_x = w / 2
    center_y = h / 2
    camera_matrix = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32) # Assume no distortion
    return camera_matrix, dist_coeffs

def load_calibration(filepath):
    """Load camera matrix and distortion coefficients from a .npz file"""
    try:
        with np.load(filepath) as X:
            camera_matrix, dist_coeffs = [X[i] for i in ('camera_matrix','dist_coeffs')]
            return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Warning: Could not load calibration data from '{filepath}': {e}")
        return None, None

def estimate_pose(corners, marker_size, camera_matrix, dist_coeffs):
    """
    Estimate the pose (rotation and translation) of markers using solvePnP.
    This works across both older and newer OpenCV versions.
    """
    marker_points = np.array([
        [-marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)
    
    rvecs, tvecs = [], []
    for corner in corners:
        success, rvec, tvec = cv2.solvePnP(
            marker_points, corner[0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if success:
            rvecs.append(rvec)
            tvecs.append(tvec)
        else:
            # Append None so the list length still matches the number of IDs!
            rvecs.append(None)
            tvecs.append(None)
            
    return rvecs, tvecs  # Removed np.array() cast so we can handle None

def detect_markers(image, aruco_dict, marker_size, camera_matrix=None, dist_coeffs=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    try:
        parameters = aruco.DetectorParameters()
    except AttributeError:
        parameters = aruco.DetectorParameters_create()
        
    try:
        # OpenCV 4.7+
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    except AttributeError:
        # Older OpenCV versions
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
        # Build camera matrix if not provided
        if camera_matrix is None or dist_coeffs is None:
            camera_matrix, dist_coeffs = get_dummy_camera_matrix(image.shape)
            
        rvecs, tvecs = estimate_pose(corners, marker_size, camera_matrix, dist_coeffs)
        
        for i in range(len(ids)):
            if tvecs[i] is None:
                continue  # Skip this marker if pose estimation failed

            c = corners[i][0]
            tvec = tvecs[i]
            rvec = rvecs[i]
            
            # Calculate distance using Euclidean norm of the translation vector
            distance = np.linalg.norm(tvec)
            
            # Formulate text: ID and Distance in meters
            info_text = f"ID: {ids[i][0]} | Dist: {distance:.2f}m"
            cv2.putText(image, info_text, 
                        (int(c[0][0]), int(c[0][1]) - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
                        
            # Draw frame axes for the marker
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, marker_size * 0.5)
            
    return image, corners, ids

def process_image(image_path, aruco_dict, marker_size, camera_matrix=None, dist_coeffs=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    result, corners, ids = detect_markers(image, aruco_dict, marker_size, camera_matrix, dist_coeffs)
    
    if ids is not None:
        print(f"Detected {len(ids)} markers.")
    else:
        print("No markers detected.")
        
    h, w = result.shape[:2]
    max_h, max_w = 800, 1200
    if h > max_h or w > max_w:
        scale = min(max_h / h, max_w / w)
        result = cv2.resize(result, (int(w * scale), int(h * scale)))
        
    cv2.imshow("ArUco Detection Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_webcam(aruco_dict, marker_size, camera_matrix=None, dist_coeffs=None, camera_id=0, resolution=None, serial_conn=None):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (ID: {camera_id}).")
        return
        
    set_camera_resolution(cap, resolution)
    
    is_dropping = False
    last_drop_time = 0

    print("Starting webcam... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
        
        result, corners, ids = detect_markers(frame, aruco_dict, marker_size, camera_matrix, dist_coeffs)
        
        if ids is not None:
            cv2.putText(result, f"Detected: {len(ids)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Check for ArUco ID 1 to sequence the Drop and Rotate
            for marker_id in ids:
                if marker_id[0] == 1: # ID 001
                    if serial_conn and not is_dropping:
                        print("ArUco ID 1 detected! Sending DROP command ('D') to Arduino...")
                        try:
                            serial_conn.write(b'D')
                            is_dropping = True
                            last_drop_time = time.time()
                        except Exception as e:
                            print(f"Serial write error: {e}")

        # Cooldown timer to prevent spamming 'D' while robot is busy dropping & rotating
        if is_dropping:
            if time.time() - last_drop_time > 5.0: # 5 second cooldown
                is_dropping = False
                print("Ready for next sequence...")
        
        cv2.imshow("Live ArUco Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Detect ArUco / AprilTag markers and estimate distance.")
    parser.add_argument(
        "--family", default="DICT_APRILTAG_36H11", 
        choices=list(FAMILIES.keys()),
        help="ArUco dictionary family (default: DICT_APRILTAG_36H11)"
    )
    parser.add_argument(
        "--image", type=str, 
        help="Path to an image file to process. If not provided, webcam will be used."
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera ID to use for webcam detection (default: 0)"
    )
    parser.add_argument(
        "--resolution", type=str, default="640x480",
        help="Target resolution for webcam (e.g., '640x480'). Must match calibration resolution!"
    )
    parser.add_argument(
        "--marker-size", type=float, default=0.2,
        help="Actual physical size of the ArUco marker in meters (default: 0.2 meters / 200 mm)."
    )
    parser.add_argument(
        "--calibration", type=str, default=None,
        help="Path to camera calibration .npz file (generated by calibrate_camera.py)."
    )
    parser.add_argument(
        "--port", type=str, default=None,
        help="Serial port for Arduino communication (e.g., COM3 or /dev/ttyUSB0). If provided, enables sending commands."
    )
    parser.add_argument(
        "--baudrate", type=int, default=115200,
        help="Serial baudrate (default: 115200)."
    )
    
    args = parser.parse_args()

    print(f"Using dictionary family: {args.family}")
    print(f"Assuming physical marker size: {args.marker_size}m")
    
    camera_matrix = None
    dist_coeffs = None
    if args.calibration:
        if os.path.exists(args.calibration):
            print(f"Loading calibration data from: {os.path.abspath(args.calibration)}")
            camera_matrix, dist_coeffs = load_calibration(args.calibration)
        else:
            print(f"Calibration file '{args.calibration}' not found. Falling back to dummy calibration.")

    aruco_dict = get_aruco_dict(args.family)
    
    # Initialize Serial Connection
    serial_conn = None
    if args.port:
        try:
            print(f"Connecting to Arduino on {args.port} at {args.baudrate} baud...")
            serial_conn = serial.Serial(args.port, args.baudrate, timeout=1)
            time.sleep(2) # Give Arduino time to reset
            print("Serial connection established.")
        except Exception as e:
            print(f"Failed to connect to serial port {args.port}: {e}")

    if args.image:
        print(f"Processing image: {args.image}")
        process_image(args.image, aruco_dict, args.marker_size, camera_matrix, dist_coeffs)
    else:
        process_webcam(aruco_dict, args.marker_size, camera_matrix, dist_coeffs, args.camera, args.resolution, serial_conn)
        
    if serial_conn and serial_conn.is_open:
        serial_conn.close()

if __name__ == "__main__":
    main()
