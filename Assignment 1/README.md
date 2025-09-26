# Computer Vision Application - Assignment 1

Muhammad Fahad Waqar<br>
st125981<br>

## Setup and Installation

# Install Dependencies:
pip install opencv-python numpy scipy matplotlib

# Run the Application:
python app.py

# Controls:

Press 1 - RGB to Gray<br>
Press 2 - RGB to HSV<br>
Press 3 - Adjust brightness (+/- keys)<br>
Press 4 - Gaussian filter<br>
Press 5 - Bilateral filter<br>
Press 6 - Canny edge detection<br>
Press 7 - Hough line detection<br>
Press 8 - Add frame for panorama<br>
Press 9 - Create panorama<br>
Press 0 - Augmented Reality<br>
Press c - Calibrate camera<br>
Press h - Show histogram<br>
Press r - Reset to normal view<br>
Press t - Image transformations<br>
Press s - Save current frame<br>
Press ESC - Exit application<br>

## Important Notes:

For AR mode: Press c to calibrate camera first using a 9x6 chessboard pattern<br>
T-Rex model: Place your .obj file at assets/trex_model.obj or the app will use a default 3D shape<br>
ArUco markers: Print DICT_6X6_250 markers for AR testing<br>
Panorama: Press 8 multiple times to capture frames, then 9 to stitch them together<br>