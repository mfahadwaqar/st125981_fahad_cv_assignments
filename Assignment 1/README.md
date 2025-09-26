# Computer Vision Application - Assignment 1

Muhammad Fahad Waqar<br>
st125981<br>

## Setup and Installation

# Install Dependencies:
pip install opencv-python numpy scipy matplotlib

# Run the Application:
python app.py

# Controls:

Key 1 - RGB to Gray<br>
Key 2 - RGB to HSV<br>
Key 3 - Adjust brightness (+/- keys)<br>
Key 4 - Gaussian filter<br>
Key 5 - Bilateral filter<br>
Key 6 - Canny edge detection<br>
Key 7 - Hough line detection<br>
Key 8 - Add frame for panorama<br>
Key 9 - Create panorama<br>
Key 0 - Augmented Reality<br>
Key c - Calibrate camera<br>
Key h - Show histogram<br>
Key r - Reset to normal view<br>
Key t - Image transformations<br>
Key s - Save current frame<br>
Key ESC - Exit application<br>

## Important Notes:

For AR mode: Press c to calibrate camera first using a 9x6 chessboard pattern<br>
T-Rex model: Place your .obj file at assets/trex_model.obj or the app will use a default 3D shape<br>
ArUco markers: Print DICT_6X6_250 markers for AR testing<br>
Panorama: Press 8 multiple times to capture frames, then 9 to stitch them together<br>