import argparse
import glob
import os

import cv2
import numpy as np
import cv2.aruco as aruco

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

def get_marker_obj_points(marker_size):
    """Return the 3D local coordinates of the marker corners."""
    s = marker_size / 2.0
    return np.array([
        [-s,  s, 0],
        [ s,  s, 0],
        [ s, -s, 0],
        [-s, -s, 0]
    ], dtype=np.float32)

def set_camera_resolution(cap, resolution_str):
    if resolution_str:
        try:
            w, h = map(int, resolution_str.lower().split('x'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Requested HD resolution {w}x{h}, webcam set to: {int(actual_w)}x{int(actual_h)}")
        except Exception as e:
            print(f"Warning: Could not set camera resolution. Error: {e}")

def detect_target_marker(image, aruco_dict, target_id):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        parameters = aruco.DetectorParameters()
    except AttributeError:
        parameters = aruco.DetectorParameters_create()
        
    try:
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    except AttributeError:
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
    if ids is not None:
        for i, marker_id in enumerate(ids):
            if marker_id[0] == target_id:
                return True, corners[i][0]
    return False, None

def calibrate_from_images(image_dir, ext, aruco_dict, target_id, marker_size):
    objp = get_marker_obj_points(marker_size)
    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(image_dir, f'*.{ext}'))
    if not images:
        print(f"No images found in {image_dir} with extension {ext}")
        return None, None

    print(f"Found {len(images)} images. Processing...")
    
    gray_shape = None
    for fname in images:
        img = cv2.imread(fname)
        if gray_shape is None:
            gray_shape = img.shape[:2][::-1]

        found, corners = detect_target_marker(img, aruco_dict, target_id)

        if found:
            objpoints.append(objp)
            
            # Subpixel refinement
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw
            cv2.polylines(img, [np.int32(corners2)], True, (0, 255, 0), 2)
            cv2.imshow('Calibration - Finding ArUco Marker', img)
            cv2.waitKey(200)
            
    cv2.destroyAllWindows()
    
    if not objpoints:
        print(f"Could not find ArUco Marker ID {target_id} in any images.")
        return None, None
        
    print(f"Successfully processed {len(objpoints)} out of {len(images)} images.")
    print("Calculating calibration parameters...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([np.array([obj]) for obj in objpoints], [np.array([img]) for img in imgpoints], gray_shape, None, None)
    
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(np.array([objpoints[i]]), rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(np.array([imgpoints[i]]), imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"Total error: {mean_error/len(objpoints)} (closer to 0 is better)")
    
    return mtx, dist

def capture_and_calibrate(camera_id, aruco_dict, target_id, marker_size, num_captures, resolution):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam ID {camera_id}.")
        return None, None

    set_camera_resolution(cap, resolution)

    objp = get_marker_obj_points(marker_size)
    objpoints = []
    imgpoints = []
    
    print("=" * 60)
    print(f"Webcam Calibration via ArUco Marker ID {target_id}")
    print(f"Note: Since a single marker only provides 4 points per image,")
    print(f"you need MANY captures from various angles/distances.")
    print(f"\nGoal: {num_captures} captures")
    print(f"Press 'c' to capture a frame with the marker visible.")
    print("Press 'q' or 'ESC' to quit early.")
    print("=" * 60)
    
    gray_shape = None
    captures = 0
    
    while captures < num_captures:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        if gray_shape is None:
            gray_shape = frame.shape[:2][::-1]
            
        display_frame = frame.copy()
        
        found, corners = detect_target_marker(frame, aruco_dict, target_id)
        if found:
            cv2.polylines(display_frame, [np.int32(corners)], True, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Marker ID {target_id} Detected", (corners[0][0], corners[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            status = "Ready to Capture ('c')"
            color = (0, 255, 0)
        else:
            status = f"Marker ID {target_id} not found"
            color = (0, 0, 255)
            
        cv2.putText(display_frame, f"Captures: {captures}/{num_captures}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Webcam Camera Calibration", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c') and found:
            objpoints.append(objp)
            
            # Subpixel refinement
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            captures += 1
            print(f"Captured {captures}/{num_captures}")
            
            # Flash screen green
            flash = np.full(frame.shape, (0, 255, 0), dtype=np.uint8)
            cv2.imshow("Webcam Camera Calibration", cv2.addWeighted(frame, 0.5, flash, 0.5, 0))
            cv2.waitKey(200)

    cap.release()
    cv2.destroyAllWindows()
    
    if captures == 0:
        print("No captures taken. Calibration aborted.")
        return None, None
        
    print(f"Calculating calibration parameters using {captures} captures...")
    
    # We must wrap objpoints and imgpoints in lists of np arrays because cv2.calibrateCamera expects a list of arrays (one for each view)
    formatted_objpoints = [np.array([obj]) for obj in objpoints]
    formatted_imgpoints = [np.array([img]) for img in imgpoints]
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(formatted_objpoints, formatted_imgpoints, gray_shape, None, None)
    
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(formatted_objpoints)):
        imgpoints2, _ = cv2.projectPoints(formatted_objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(formatted_imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"Total error: {mean_error/len(formatted_objpoints)} (closer to 0 is better)")
    
    return mtx, dist

def main():
    parser = argparse.ArgumentParser(description="Calibrate camera using a single ArUco marker.")
    parser.add_argument("--mode", choices=["webcam", "images"], default="webcam", 
                        help="Calibration mode: capture live from 'webcam' or load 'images' from a folder.")
    parser.add_argument("--dir", default="calibration_images", 
                        help="Directory containing calibration images (used in 'images' mode).")
    parser.add_argument("--ext", default="jpg", 
                        help="Image extension to search for in 'images' mode (e.g., jpg, png).")
    parser.add_argument("--family", default="DICT_APRILTAG_36H11", 
                        choices=list(FAMILIES.keys()),
                        help="ArUco dictionary family (default: DICT_APRILTAG_36H11)")
    parser.add_argument("--marker-id", type=int, default=0,
                        help="The ID of the ArUco marker to use for calibration (default: 0)")
    parser.add_argument("--marker-size", type=float, default=0.2, 
                        help="Size of the ArUco marker in meters (default: 0.2m / 200mm).")
    parser.add_argument("--camera", type=int, default=0, 
                        help="Camera ID for webcam mode (change to 1, 2, etc. for external HD webcams).")
    parser.add_argument("--resolution", type=str, default="1920x1080", 
                        help="Target resolution for webcam (e.g., '1920x1080', '1280x720').")
    parser.add_argument("--captures", type=int, default=40, 
                        help="Number of images to capture in webcam mode. (Recommend 40+ for single-marker).")
    parser.add_argument("--output", default="camera_calibration.npz", 
                        help="Output file to save camera matrix and distance coefficients.")
    
    args = parser.parse_args()
    
    print(f"Calibration Configuration:")
    print(f" - ArUco Family : {args.family}")
    print(f" - Target ID    : {args.marker_id}")
    print(f" - Marker Size  : {args.marker_size} meters")
    
    aruco_dict = get_aruco_dict(args.family)
    
    if args.mode == "images":
        mtx, dist = calibrate_from_images(args.dir, args.ext, aruco_dict, args.marker_id, args.marker_size)
    else:
        mtx, dist = capture_and_calibrate(args.camera, aruco_dict, args.marker_id, args.marker_size, args.captures, args.resolution)
        
    if mtx is not None and dist is not None:
        np.savez(args.output, camera_matrix=mtx, dist_coeffs=dist)
        print(f"Calibration data saved to: {os.path.abspath(args.output)}")
        print("You can now use this with detect_arucos.py:")
        print(f"python detect_arucos.py --calibration \"{args.output}\"")

if __name__ == "__main__":
    main()
