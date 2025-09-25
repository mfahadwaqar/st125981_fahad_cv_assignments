# Muhammad Fahad Waqar
# st125981
# Assignment No. 1

import cv2
import numpy as np
import argparse
import os
from scipy import ndimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time

class ComputerVisionApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Front camera
        self.calibrated = False
        self.camera_matrix = None
        self.dist_coeffs = None
        self.panorama_images = []
        self.trex_vertices = None
        self.trex_faces = None

        # Try loading saved calibration file
        if os.path.exists("camera_calibration.npz"):
            data = np.load("camera_calibration.npz")
            self.camera_matrix = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]
            self.calibrated = True
            print("Loaded saved camera calibration from file.")
        else:
            print("No saved camera calibration found. Run calibration first.")

        # ArUco detector setup - Fixed for newer OpenCV versions
        try:
            # Try new API first (OpenCV 4.7+)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            try:
                # Try intermediate API (OpenCV 4.0-4.6)
                self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            except AttributeError:
                # Fallback for very old versions
                self.aruco_dict = cv2.aruco.dict_get(cv2.aruco.DICT_6X6_250)
                self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Load TREX model
        self.load_trex_model()
        
    def load_trex_model(self):
        """Load the T-Rex OBJ model"""
        try:
            vertices = []
            faces = []
            
            # Try different possible locations for the T-Rex model
            possible_paths = [
                'assets/trex_model.obj',  # Your actual path first
                'trex_model.obj',
                'trex.obj',
                'models/trex_model.obj',
                'models/trex.obj',
                'assets/trex.obj'
            ]
            
            model_loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as file:
                        for line in file:
                            if line.startswith('v '):
                                parts = line.strip().split()
                                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                                vertices.append(vertex)
                            elif line.startswith('f '):
                                parts = line.strip().split()
                                face = []
                                for part in parts[1:]:
                                    face.append(int(part.split('/')[0]) - 1)  # OBJ is 1-indexed
                                faces.append(face)
                    
                    self.trex_vertices = np.array(vertices, dtype=np.float32) * 20  # Smaller scale
                    self.trex_faces = faces
                    print(f"Loaded T-Rex model from {path} with {len(vertices)} vertices and {len(faces)} faces")
                    model_loaded = True
                    break
            
            if not model_loaded:
                print("T-Rex model file not found, using default 3D dinosaur shape")
                self.create_default_trex()
        except Exception as e:
            print(f"Error loading T-Rex model: {e}")
            self.create_default_trex()
    
    # 1. Color conversion functions
    def rgb_to_gray(self, image):
        """Convert RGB to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def rgb_to_hsv(self, image):
        """Convert RGB to HSV"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    def hsv_to_rgb(self, image):
        """Convert HSV to RGB"""
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    
    def gray_to_rgb(self, image):
        """Convert grayscale to RGB"""
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 2. Contrast and brightness adjustment
    def adjust_contrast_brightness(self, image, alpha=1.0, beta=0):
        """Adjust contrast (alpha) and brightness (beta)"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 3. Histogram calculation
    def calculate_histogram(self, image):
        """Calculate and display histogram"""
        if len(image.shape) == 3:
            # Color image
            colors = ('b', 'g', 'r')
            plt.figure(figsize=(10, 4))
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=color)
            plt.title('Color Histogram')
            plt.xlabel('Bins')
            plt.ylabel('# of Pixels')
            plt.show()
        else:
            # Grayscale image
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            plt.figure(figsize=(10, 4))
            plt.plot(hist)
            plt.title('Grayscale Histogram')
            plt.xlabel('Bins')
            plt.ylabel('# of Pixels')
            plt.show()
        return hist

    # 4. Gaussian filter
    def gaussian_filter(self, image, kernel_size=5, sigma=1.0):
        """Apply Gaussian filter with changeable parameters"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # 5. Bilateral filter
    def bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter with changeable parameters"""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    # 6. Canny edge detection
    def canny_edge_detection(self, image, low_threshold=50, high_threshold=150):
        """Apply Canny edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Canny(gray, low_threshold, high_threshold)

    # 7. Hough line detection
    def hough_line_detection(self, image):
        """Detect lines using Hough Transform"""
        edges = self.canny_edge_detection(image)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=50, maxLineGap=10)
        
        result = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return result, lines

    # 8. Panorama creation (keeping original implementation)
    def create_panorama(self, images):
        """Create panorama from multiple images - completely from scratch implementation"""
        if len(images) < 2:
            return None
        
        # Initialize with first image
        result = images[0]
        
        for i in range(1, len(images)):
            result = self.stitch_images(result, images[i])
            if result is None:
                print(f"Failed to stitch image {i}")
                return None
        
        return result
    
    def stitch_images(self, img1, img2):
        """Stitch two images together using completely custom implementation"""
        # Convert to grayscale
        gray1 = self.rgb_to_gray(img1)
        gray2 = self.rgb_to_gray(img2)
        
        # Custom feature detection using Harris corners
        corners1 = self.harris_corner_detection(gray1)
        corners2 = self.harris_corner_detection(gray2)
        
        if len(corners1) < 10 or len(corners2) < 10:
            print("Not enough corner features detected")
            return None
        
        # Custom feature matching
        matches = self.match_features(gray1, gray2, corners1, corners2)
        
        if len(matches) < 8:
            print("Not enough feature matches found")
            return None
        
        # Custom homography estimation using RANSAC
        H = self.estimate_homography_ransac(matches)
        
        if H is None:
            print("Failed to estimate homography")
            return None
        
        # Custom image warping and blending
        result = self.warp_and_blend_images(img1, img2, H)
        
        return result
    
    def harris_corner_detection(self, image, k=0.04, threshold=0.01):
        """Custom Harris corner detection implementation"""
        # Compute gradients
        Ix = self.compute_gradient_x(image)
        Iy = self.compute_gradient_y(image)
        
        # Compute products of derivatives
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy
        
        # Apply Gaussian smoothing
        Ixx = self.gaussian_smooth(Ixx, sigma=1.0)
        Ixy = self.gaussian_smooth(Ixy, sigma=1.0)
        Iyy = self.gaussian_smooth(Iyy, sigma=1.0)
        
        # Compute Harris response
        det = Ixx * Iyy - Ixy * Ixy
        trace = Ixx + Iyy
        harris_response = det - k * (trace * trace)
        
        # Find corners above threshold
        corners = []
        h, w = harris_response.shape
        for y in range(3, h-3):
            for x in range(3, w-3):
                if (harris_response[y, x] > threshold and 
                    harris_response[y, x] == np.max(harris_response[y-3:y+4, x-3:x+4])):
                    corners.append((x, y))
        
        return corners[:500]  # Limit to top 500 corners
    
    def compute_gradient_x(self, image):
        """Compute horizontal gradient using Sobel operator"""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        return self.convolve2d(image, sobel_x)
    
    def compute_gradient_y(self, image):
        """Compute vertical gradient using Sobel operator"""
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        return self.convolve2d(image, sobel_y)
    
    def convolve2d(self, image, kernel):
        """Custom 2D convolution implementation"""
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        # Pad image
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        result = np.zeros_like(image, dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                result[y, x] = np.sum(padded[y:y+kh, x:x+kw] * kernel)
        
        return result
    
    def gaussian_smooth(self, image, sigma=1.0, kernel_size=5):
        """Custom Gaussian smoothing"""
        kernel = self.gaussian_kernel(kernel_size, sigma)
        return self.convolve2d(image, kernel)
    
    def gaussian_kernel(self, size, sigma):
        """Generate Gaussian kernel"""
        kernel = np.zeros((size, size))
        center = size // 2
        sum_val = 0
        
        for y in range(size):
            for x in range(size):
                diff_x = x - center
                diff_y = y - center
                val = np.exp(-(diff_x*diff_x + diff_y*diff_y) / (2 * sigma * sigma))
                kernel[y, x] = val
                sum_val += val
        
        return kernel / sum_val
    
    def match_features(self, img1, img2, corners1, corners2, window_size=15):
        """Custom feature matching using normalized cross-correlation"""
        matches = []
        half_window = window_size // 2
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        
        for i, (x1, y1) in enumerate(corners1):
            if (x1 < half_window or x1 >= w1 - half_window or 
                y1 < half_window or y1 >= h1 - half_window):
                continue
            
            patch1 = img1[y1-half_window:y1+half_window+1, x1-half_window:x1+half_window+1]
            patch1 = patch1.astype(np.float32)
            patch1 = (patch1 - np.mean(patch1)) / (np.std(patch1) + 1e-8)
            
            best_match = -1
            best_score = -1
            
            for j, (x2, y2) in enumerate(corners2):
                if (x2 < half_window or x2 >= w2 - half_window or 
                    y2 < half_window or y2 >= h2 - half_window):
                    continue
                
                patch2 = img2[y2-half_window:y2+half_window+1, x2-half_window:x2+half_window+1]
                patch2 = patch2.astype(np.float32)
                patch2 = (patch2 - np.mean(patch2)) / (np.std(patch2) + 1e-8)
                
                # Normalized cross-correlation
                ncc = np.sum(patch1 * patch2) / (window_size * window_size)
                
                if ncc > best_score:
                    best_score = ncc
                    best_match = j
            
            if best_score > 0.5:  # Threshold for good matches
                matches.append(((x1, y1), corners2[best_match], best_score))
        
        # Sort by score and return best matches
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:50]  # Return top 50 matches
    
    def estimate_homography_ransac(self, matches, max_iterations=1000, threshold=3.0):
        """Custom RANSAC homography estimation"""
        if len(matches) < 8:
            return None
        
        best_H = None
        best_inliers = 0
        
        for _ in range(max_iterations):
            # Randomly select 4 matches
            sample_matches = np.random.choice(len(matches), 4, replace=False)
            sample_points = [matches[i] for i in sample_matches]
            
            # Compute homography from 4 points
            H = self.compute_homography_4points(sample_points)
            
            if H is None:
                continue
            
            # Count inliers
            inliers = 0
            for match in matches:
                pt1, pt2, _ = match
                # Transform point using homography
                transformed = self.transform_point(pt1, H)
                if transformed is not None:
                    distance = np.sqrt((transformed[0] - pt2[0])**2 + (transformed[1] - pt2[1])**2)
                    if distance < threshold:
                        inliers += 1
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H
        
        return best_H if best_inliers >= 8 else None
    
    def compute_homography_4points(self, matches):
        """Compute homography from 4 point correspondences"""
        if len(matches) != 4:
            return None
        
        # Build system of equations Ah = 0
        A = []
        for match in matches:
            (x1, y1), (x2, y2), _ = match
            A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
            A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
        
        A = np.array(A, dtype=np.float32)
        
        # Solve using SVD
        try:
            U, S, Vt = np.linalg.svd(A)
            h = Vt[-1, :]
            H = h.reshape(3, 3)
            return H / H[2, 2] if H[2, 2] != 0 else H
        except:
            return None
    
    def transform_point(self, point, H):
        """Transform a point using homography matrix"""
        x, y = point
        homogeneous = np.array([x, y, 1])
        transformed = H @ homogeneous
        
        if abs(transformed[2]) < 1e-8:
            return None
        
        return (transformed[0] / transformed[2], transformed[1] / transformed[2])
    
    def warp_and_blend_images(self, img1, img2, H):
        """Custom image warping and blending"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Get corners of both images
        corners1 = [(0, 0), (w1, 0), (w1, h1), (0, h1)]
        corners2 = [(0, 0), (w2, 0), (w2, h2), (0, h2)]
        
        # Transform corners of second image
        transformed_corners2 = []
        for corner in corners2:
            transformed = self.transform_point(corner, H)
            if transformed:
                transformed_corners2.append(transformed)
        
        if len(transformed_corners2) != 4:
            return None
        
        # Calculate output image bounds
        all_corners = corners1 + transformed_corners2
        min_x = int(min(corner[0] for corner in all_corners))
        max_x = int(max(corner[0] for corner in all_corners))
        min_y = int(min(corner[1] for corner in all_corners))
        max_y = int(max(corner[1] for corner in all_corners))
        
        # Create output image
        output_w = max_x - min_x
        output_h = max_y - min_y
        result = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        
        # Translation offset
        offset_x = -min_x
        offset_y = -min_y
        
        # Place first image
        y_start = max(0, offset_y)
        y_end = min(output_h, offset_y + h1)
        x_start = max(0, offset_x)
        x_end = min(output_w, offset_x + w1)
        
        img1_y_start = max(0, -offset_y)
        img1_y_end = img1_y_start + (y_end - y_start)
        img1_x_start = max(0, -offset_x)
        img1_x_end = img1_x_start + (x_end - x_start)
        
        result[y_start:y_end, x_start:x_end] = img1[img1_y_start:img1_y_end, img1_x_start:img1_x_end]
        
        # Warp second image using inverse homography
        try:
            H_inv = np.linalg.inv(H)
        except:
            return None
        
        # Forward mapping with interpolation
        for y in range(output_h):
            for x in range(output_w):
                # Map output pixel back to second image
                src_point = self.transform_point((x - offset_x, y - offset_y), H_inv)
                
                if src_point:
                    src_x, src_y = src_point
                    if 0 <= src_x < w2-1 and 0 <= src_y < h2-1:
                        # Bilinear interpolation
                        pixel = self.bilinear_interpolate(img2, src_x, src_y)
                        if pixel is not None:
                            # Simple blending - average if both images have pixels
                            if np.sum(result[y, x]) > 0:
                                result[y, x] = (result[y, x].astype(np.float32) + pixel.astype(np.float32)) / 2
                            else:
                                result[y, x] = pixel
        
        return result
    
    def bilinear_interpolate(self, image, x, y):
        """Bilinear interpolation for sub-pixel sampling"""
        x1, x2 = int(x), int(x) + 1
        y1, y2 = int(y), int(y) + 1
        
        if x2 >= image.shape[1] or y2 >= image.shape[0]:
            return None
        
        # Get the four neighboring pixels
        I11 = image[y1, x1].astype(np.float32)
        I12 = image[y2, x1].astype(np.float32)
        I21 = image[y1, x2].astype(np.float32)
        I22 = image[y2, x2].astype(np.float32)
        
        # Compute weights
        wa = (x2 - x) * (y2 - y)
        wb = (x2 - x) * (y - y1)
        wc = (x - x1) * (y2 - y)
        wd = (x - x1) * (y - y1)
        
        # Interpolate
        interpolated = wa * I11 + wb * I12 + wc * I21 + wd * I22
        
        return np.clip(interpolated, 0, 255).astype(np.uint8)

    # 9. Image transformations
    def translate_image(self, image, tx, ty):
        """Translate image by (tx, ty)"""
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    def rotate_image(self, image, angle, center=None):
        """Rotate image by angle (degrees)"""
        if center is None:
            center = (image.shape[1]//2, image.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    def scale_image(self, image, scale_x, scale_y):
        """Scale image by scale factors"""
        M = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
        new_width = int(image.shape[1] * scale_x)
        new_height = int(image.shape[0] * scale_y)
        return cv2.warpAffine(image, M, (new_width, new_height))

    # 10. Camera calibration with automatic capture
    def calibrate_camera(self, chessboard_size=(9, 6), num_images=20, auto_capture_delay=2.0):
        """Calibrate camera using chessboard pattern with automatic capture"""
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        
        print(f"Auto-capturing {num_images} images for calibration...")
        print(f"Make sure chessboard is visible. Images will be captured automatically every {auto_capture_delay} seconds when pattern is detected.")
        print("Press ESC to finish early")
        
        captured = 0
        last_capture_time = 0
        
        while captured < num_images:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            # Display frame
            display_frame = frame.copy()
            current_time = time.time()
            
            if ret_corners:
                cv2.drawChessboardCorners(display_frame, chessboard_size, corners, ret_corners)
                
                # Auto capture if enough time has passed
                if current_time - last_capture_time > auto_capture_delay:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    captured += 1
                    last_capture_time = current_time
                    print(f"Auto-captured image {captured}/{num_images}")
                    cv2.putText(display_frame, f"CAPTURED! ({captured}/{num_images})", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                cv2.putText(display_frame, f"Pattern detected! ({captured}/{num_images})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No pattern detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        cv2.destroyWindow('Camera Calibration')
        
        if len(objpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
            
            if ret:
                self.camera_matrix = mtx
                self.dist_coeffs = dist
                self.calibrated = True
                print("Camera calibration successful!")
                print(f"Camera matrix:\n{mtx}")
                print(f"Distortion coefficients: {dist}")

                # Save calibration
                np.savez('camera_calibration.npz', camera_matrix=mtx, dist_coeffs=dist)
                print("Saved calibration to camera_calibration.npz")

                return True
            else:
                print("Camera calibration failed!")
                return False
        else:
            print("No valid calibration images captured!")
            return False

    # 11. Augmented Reality with ArUco markers and T-Rex model
    def augmented_reality(self, image):
        """Augmented reality with T-Rex model using ArUco markers - Fixed version"""
        if not self.calibrated:
            print("Camera not calibrated! Run calibration first.")
            return image
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers - Fixed for newer OpenCV versions
        try:
            # Try new API first (OpenCV 4.7+)
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
        except AttributeError:
            try:
                # Try intermediate API (OpenCV 4.0-4.6)
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            except AttributeError:
                # Fallback for very old versions
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None and len(corners) > 0:
            # Draw detected markers
            try:
                result = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
            except:
                result = image.copy()
                cv2.aruco.drawDetectedMarkers(result, corners, ids)
            
            # For each detected marker, estimate pose and draw T-Rex
            for i, corner in enumerate(corners):
                # Define marker size in real world (e.g., 5cm = 0.05m)
                marker_size = 0.05
                
                # Estimate pose using solvePnP (most reliable method)
                try:
                    # Create 3D object points for a square marker
                    half_size = marker_size / 2
                    object_points = np.array([
                        [-half_size, -half_size, 0],
                        [half_size, -half_size, 0],
                        [half_size, half_size, 0],
                        [-half_size, half_size, 0]
                    ], dtype=np.float32)
                    
                    # Use the corner points directly
                    image_points = corner[0].astype(np.float32)
                    
                    success, rvec, tvec = cv2.solvePnP(
                        object_points, image_points, 
                        self.camera_matrix, self.dist_coeffs)
                    
                    if success:
                        # Draw coordinate axes on marker
                        try:
                            cv2.drawFrameAxes(result, self.camera_matrix, self.dist_coeffs, 
                                            rvec, tvec, marker_size)
                        except Exception as axis_error:
                            print(f"Axis drawing warning (can be ignored): {axis_error}")
                        
                        # Draw T-Rex model above the marker
                        result = self.draw_trex_on_marker(result, rvec, tvec)
                        
                except Exception as e:
                    print(f"Pose estimation failed: {e}")
                    continue
            
            return result
        
        return image
        
    def draw_trex_on_marker(self, image, rvec, tvec):
        """Draw the T-Rex model on the detected ArUco marker - Fixed version"""
        try:
            # Ensure rvec and tvec are properly shaped
            rvec = rvec.reshape(-1)
            tvec = tvec.reshape(-1)
            
            # Scale the T-Rex model appropriately
            scaled_trex_vertices = self.trex_vertices * 0.008  # Increased scale for visibility
            
            # Create offset to place T-Rex above the marker
            tvec_offset = tvec.copy()
            tvec_offset[2] -= 0.02  # Move 2cm above marker (negative Z is up in camera coords)
            
            # Project T-Rex vertices to image coordinates
            projected_points, jacobian = cv2.projectPoints(
                scaled_trex_vertices, rvec, tvec_offset, 
                self.camera_matrix, self.dist_coeffs)
            
            # Convert to integer coordinates and reshape
            projected_points = projected_points.reshape(-1, 2)
            
            # Filter out invalid points (NaN or extremely large values)
            valid_mask = (
                np.isfinite(projected_points[:, 0]) & 
                np.isfinite(projected_points[:, 1]) &
                (np.abs(projected_points[:, 0]) < 10000) &
                (np.abs(projected_points[:, 1]) < 10000)
            )
            
            if np.any(valid_mask):
                # Only keep valid points
                valid_projected_points = projected_points[valid_mask]
                
                # Create mapping from original indices to valid indices
                valid_indices = np.where(valid_mask)[0]
                index_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_indices)}
                
                # Convert to integer coordinates
                valid_projected_points = valid_projected_points.astype(np.int32)
                
                # Draw T-Rex model
                result = self.draw_trex_model_safe(image, valid_projected_points, 
                                                valid_indices, index_mapping)
                return result
            
            return image
            
        except Exception as e:
            print(f"Error drawing T-Rex: {e}")
            return image
    
    def draw_trex_model_safe(self, image, projected_points, valid_indices, index_mapping):
        """Draw the T-Rex model using projected points with improved safety checks"""
        result = image.copy()
        h, w = image.shape[:2]
        
        try:
            # Draw faces of the T-Rex model
            for face in self.trex_faces:
                if len(face) >= 3:
                    # Check if all vertices of this face are valid
                    face_points = []
                    valid_face = True
                    
                    for vertex_idx in face:
                        if vertex_idx in index_mapping:
                            mapped_idx = index_mapping[vertex_idx]
                            if mapped_idx < len(projected_points):
                                x, y = projected_points[mapped_idx]
                                # Check if point is within reasonable screen bounds
                                if 0 <= x < w and 0 <= y < h:
                                    face_points.append((int(x), int(y)))
                                else:
                                    # Allow some points slightly outside screen
                                    if -w//2 < x < w*1.5 and -h//2 < y < h*1.5:
                                        face_points.append((int(np.clip(x, 0, w-1)), 
                                                        int(np.clip(y, 0, h-1))))
                                    else:
                                        valid_face = False
                                        break
                            else:
                                valid_face = False
                                break
                        else:
                            valid_face = False
                            break
                    
                    # Draw the face if it's valid and has enough points
                    if valid_face and len(face_points) >= 3:
                        pts = np.array(face_points, np.int32)
                        
                        # Simple depth-based coloring (darker for faces further back)
                        # Calculate average Z coordinate for this face
                        face_z = 0
                        valid_z_count = 0
                        for vertex_idx in face:
                            if vertex_idx < len(self.trex_vertices):
                                face_z += self.trex_vertices[vertex_idx][2]
                                valid_z_count += 1
                        
                        if valid_z_count > 0:
                            avg_z = face_z / valid_z_count
                            # Normalize brightness based on Z coordinate
                            brightness = max(0.3, min(1.0, (avg_z + 2) / 4))  # Adjust range as needed
                        else:
                            brightness = 0.7
                        
                        # Color the T-Rex green with varying brightness
                        color = (int(20 * brightness), int(180 * brightness), int(20 * brightness))
                        outline_color = (0, 255, 0)
                        
                        # Draw filled face and outline
                        cv2.fillPoly(result, [pts], color)
                        cv2.polylines(result, [pts], True, outline_color, 1)
            
            return result
            
        except Exception as e:
            print(f"Error in draw_trex_model_safe: {e}")
            return image
    
    def create_default_trex(self):
        """Create a more visible default T-Rex-like shape"""
        # Create a simpler, more visible T-Rex shape
        self.trex_vertices = np.array([
            # Main body (rectangular prism)
            [-2.0, -1.5, 0], [2.0, -1.5, 0], [2.0, 1.5, 0], [-2.0, 1.5, 0],  # bottom face
            [-2.0, -1.5, 3.0], [2.0, -1.5, 3.0], [2.0, 1.5, 3.0], [-2.0, 1.5, 3.0],  # top face
            
            # Head (front extension)
            [2.0, -1.0, 2.5], [3.5, -1.0, 2.5], [3.5, 1.0, 2.5], [2.0, 1.0, 2.5],  # head bottom
            [2.0, -1.0, 4.0], [3.5, -1.0, 4.0], [3.5, 1.0, 4.0], [2.0, 1.0, 4.0],  # head top
            
            # Tail (back extension)
            [-2.0, -0.8, 1.5], [-4.0, -0.5, 1.0], [-4.0, 0.5, 1.0], [-2.0, 0.8, 1.5],  # tail
            
            # Simple legs (four support pillars)
            [-1.0, -1.0, 0], [-0.5, -1.0, 0], [-0.5, -0.5, 0], [-1.0, -0.5, 0],  # leg 1 base
            [-1.0, -1.0, -2.0], [-0.5, -1.0, -2.0], [-0.5, -0.5, -2.0], [-1.0, -0.5, -2.0],  # leg 1 bottom
            
            [0.5, -1.0, 0], [1.0, -1.0, 0], [1.0, -0.5, 0], [0.5, -0.5, 0],  # leg 2 base
            [0.5, -1.0, -2.0], [1.0, -1.0, -2.0], [1.0, -0.5, -2.0], [0.5, -0.5, -2.0],  # leg 2 bottom
            
        ], dtype=np.float32)
        
        # Flip the model if it appears inverted (flip Y and Z coordinates)
        self.trex_vertices[:, 1] *= -1  # Flip Y axis
        self.trex_vertices[:, 2] *= -1  # Flip Z axis
        
        # Define faces for the simplified T-Rex
        self.trex_faces = [
            # Main body faces
            [0, 1, 2, 3],       # bottom
            [4, 7, 6, 5],       # top
            [0, 4, 5, 1],       # front
            [2, 6, 7, 3],       # back
            [0, 3, 7, 4],       # left
            [1, 5, 6, 2],       # right
            
            # Head faces
            [8, 9, 10, 11],     # bottom
            [12, 15, 14, 13],   # top
            [8, 12, 13, 9],     # front
            [10, 14, 15, 11],   # back
            [8, 11, 15, 12],    # left
            [9, 13, 14, 10],    # right
            
            # Tail faces (triangular)
            [16, 17, 18],       # triangle 1
            [16, 18, 19],       # triangle 2
            
            # Leg 1 faces
            [20, 21, 22, 23],   # top
            [24, 27, 26, 25],   # bottom
            [20, 24, 25, 21],   # side 1
            [22, 26, 27, 23],   # side 2
            [20, 23, 27, 24],   # side 3
            [21, 25, 26, 22],   # side 4
            
            # Leg 2 faces
            [28, 29, 30, 31],   # top
            [32, 35, 34, 33],   # bottom
            [28, 32, 33, 29],   # side 1
            [30, 34, 35, 31],   # side 2
            [28, 31, 35, 32],   # side 3
            [29, 33, 34, 30],   # side 4
        ]

    def run_demo(self):
        """Run interactive demo of all features"""
        print("Computer Vision App Demo")
        print("Controls:")
        print("1: RGB to Gray")
        print("2: RGB to HSV") 
        print("3: Adjust brightness (+/-)")
        print("4: Gaussian filter")
        print("5: Bilateral filter")
        print("6: Canny edge detection")
        print("7: Hough line detection")
        print("8: Add frame for panorama")
        print("9: Create panorama")
        print("0: Augmented Reality (ArUco)")
        print("c: Calibrate camera (auto-capture)")
        print("h: Show histogram")
        print("r: Reset to normal view")
        print("t: Image transformations")
        print("s: Save current frame")
        print("ESC: Quit")
        
        current_mode = 'normal'
        brightness_offset = 0
        contrast_factor = 1.0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed_frame = frame.copy()
            
            # Apply current mode
            if current_mode == 'gray':
                processed_frame = self.rgb_to_gray(frame)
                processed_frame = self.gray_to_rgb(processed_frame)
            elif current_mode == 'hsv':
                processed_frame = self.rgb_to_hsv(frame)
            elif current_mode == 'brightness':
                processed_frame = self.adjust_contrast_brightness(frame, contrast_factor, brightness_offset)
            elif current_mode == 'gaussian':
                processed_frame = self.gaussian_filter(frame, 15, 2.0)
            elif current_mode == 'bilateral':
                processed_frame = self.bilateral_filter(frame, 15, 80, 80)
            elif current_mode == 'canny':
                edges = self.canny_edge_detection(frame)
                processed_frame = self.gray_to_rgb(edges)
            elif current_mode == 'hough':
                processed_frame, _ = self.hough_line_detection(frame)
            elif current_mode == 'ar':
                processed_frame = self.augmented_reality(frame)
            elif current_mode == 'transform':
                processed_frame = self.rotate_image(frame, 15)
            
            # Display frame
            cv2.putText(processed_frame, f"Mode: {current_mode}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if current_mode == 'ar' and not self.calibrated:
                cv2.putText(processed_frame, "Press 'c' to calibrate camera first", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Computer Vision App', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('1'):
                current_mode = 'gray'
            elif key == ord('2'):
                current_mode = 'hsv'
            elif key == ord('3'):
                current_mode = 'brightness'
            elif key == ord('4'):
                current_mode = 'gaussian'
            elif key == ord('5'):
                current_mode = 'bilateral'
            elif key == ord('6'):
                current_mode = 'canny'
            elif key == ord('7'):
                current_mode = 'hough'
            elif key == ord('8'):
                self.panorama_images.append(frame.copy())
                print(f"Added frame to panorama ({len(self.panorama_images)} images)")
            elif key == ord('9'):
                if len(self.panorama_images) >= 2:
                    print("Creating panorama...")
                    pano = self.create_panorama(self.panorama_images)
                    if pano is not None:
                        cv2.imshow('Panorama', pano)
                        cv2.imwrite('panorama.jpg', pano)
                        print("Panorama saved as panorama.jpg")
                    else:
                        print("Failed to create panorama")
                else:
                    print("Need at least 2 images for panorama")
            elif key == ord('0'):
                current_mode = 'ar'
            elif key == ord('c'):
                print("Starting camera calibration...")
                self.calibrate_camera()
            elif key == ord('h'):
                self.calculate_histogram(processed_frame)
            elif key == ord('r'):
                current_mode = 'normal'
            elif key == ord('t'):
                current_mode = 'transform'
            elif key == ord('s'):
                cv2.imwrite(f'captured_frame_{current_mode}.jpg', processed_frame)
                print(f"Saved frame as captured_frame_{current_mode}.jpg")
            elif key == ord('+') or key == ord('='):
                if current_mode == 'brightness':
                    brightness_offset = min(brightness_offset + 10, 100)
            elif key == ord('-'):
                if current_mode == 'brightness':
                    brightness_offset = max(brightness_offset - 10, -100)
            elif key == 27:  # ESC
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Computer Vision Application')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    parser.add_argument('--calibrate', action='store_true', help='Calibrate camera only')
    
    args = parser.parse_args()
    
    app = ComputerVisionApp()
    
    if args.calibrate:
        app.calibrate_camera()
    else:
        app.run_demo()

if __name__ == "__main__":
    main()