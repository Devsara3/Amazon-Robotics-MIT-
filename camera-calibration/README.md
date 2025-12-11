# Camera Calibration Project

This project performs camera calibration using a chessboard pattern and OpenCV.  
The goal is to estimate the camera's intrinsic parameters and lens distortion coefficients.

---

## Project Structure

- code : Streo_vision.ipynb file has calibration script, rectification script, creating disparity maps   
- results : calibration parameters (intrinsic matrix, distortion coefficients)

---

## Results

- Mean reprojection error: **0.0424 px**
- Saved files:
  - `intrinsic.npy`
  - `distortion.npy`

---

## Camera Parameters Explained

### **1. Intrinsic Matrix (K)**  
File: `intrinsic.npy`

The intrinsic matrix describes the internal geometry of the camera.  
It contains:

K =

[ fx 0 cx

  0 fy cy

  0 0 1 ]

Where:
- \( f_x, f_y \): focal lengths in pixels  
- \( c_x, c_y \): principal point (image center)

This matrix is essential for:
- projecting 3D points to 2D,
- pose estimation (`solvePnP`),
- triangulation,
- stereo reconstruction.

---

### **2. Distortion Coefficients (dist)**  
File: `distortion.npy`

OpenCV uses a 5-parameter distortion model:

\[
[k_1,\; k_2,\; p_1,\; p_2,\; k_3]
\]

Meaning:
- \(k_1, k_2, k_3\): radial distortion  
  (typical barrel / pincushion shape from the lens)
- \(p_1, p_2\): tangential distortion  
  (caused by lens misalignment)

These coefficients allow the camera images to be **undistorted** so straight lines appear straight.

---

## Usage

Load the parameters:

```python
import numpy as np

K = np.load("results/intrinsic.npy")
dist = np.load("results/distortion.npy")
