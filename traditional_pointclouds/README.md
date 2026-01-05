
# Stereo Camera Calibration, Disparity & Point Cloud Pipeline (Google Colab)

## Overview

This repository contains a **complete stereo vision pipeline** implemented with **Google Colab notebooks**.
The pipeline covers the entire process from **stereo camera calibration** to **disparity estimation** and finally **3D point cloud generation**.

The workflow is designed to be modular but sequential, allowing each stage to be verified independently while also supporting a full end-to-end execution.

All calibration steps explicitly use the checkerboard images stored in the **`chess` folder**.

---

## Pipeline Flow (End-to-End)

The full pipeline proceeds as follows:

1. **Stereo Calibration Step 1** (`stereo_calib1.ipynb`)
2. **Stereo Calibration Step 2** (`stereo_calib2.ipynb`)
3. **Stereo Calibration Step 3** (`stereo_calib3.ipynb`)
4. **Rectification & Undistortion** (`rectify_and_undistorted.ipynb`)
5. **Point Cloud Generation**

   * `pointclouds_before.ipynb`
   * `pointclouds_improve.ipynb`

After completing these steps, **3D point clouds can be generated and exported**.

---

## Calibration Images 

* Camera calibration **always uses the images inside the `chess` folder**.
* The images contain a **7×7 internal-corner checkerboard** captured from:

  * Multiple viewing angles
  * Different distances
* These images are required for:

  * Intrinsic parameter estimation
  * Stereo parameter estimation
  * Reprojection error evaluation

---

## Notebooks Description

| Notebook                        | Role                   | Description                                                                                                                   |
| ------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `stereo_calib1.ipynb`           | Calibration (Step 1)   | Detects checkerboard corners and computes initial intrinsic parameters for left and right cameras.                            |
| `stereo_calib2.ipynb`           | Calibration (Step 2)   | Refines intrinsic and stereo parameters, evaluates reprojection errors, and filters unstable images.                          |
| `stereo_calib3.ipynb`           | Calibration (Step 3)   | Finalizes stereo calibration parameters used for rectification and depth estimation.                                          |
| `rectify_and_undistorted.ipynb` | Rectification          | Performs stereo rectification and undistortion using finalized calibration results. Produces aligned stereo image pairs.      |
| `pointclouds_before.ipynb`      | Point Cloud (Baseline) | Generates a **simple point cloud** directly from disparity maps and rectified images.                                         |
| `pointclouds_improve.ipynb`     | Point Cloud (Improved) | Generates **improved point clouds**, handling texture-less regions using methods such as **RANSAC** and additional filtering. |

---

## Point Cloud Generation (Two Versions)

### 1. Point Clouds – *Before* (Baseline)

* Simple and direct point cloud generation
* Uses:

  * Rectified stereo images
  * Disparity maps
* Suitable for:

  * Well-textured surfaces
  * Basic 3D reconstruction verification

### 2. Point Clouds – *Improve* (Enhanced)

* Designed to improve reconstruction quality in **texture-less or noisy regions**
* Applies:

  * RANSAC-based plane fitting
  * Additional filtering and robustness techniques
* Produces:

  * More stable and visually consistent point clouds
  * Reduced noise and outliers

---

## Usage (Google Colab)

1. Open notebooks in **Google Colab**.
2. Mount Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Ensure the following structure exists:

   * `chess/` (checkerboard images)
   * Stereo image folders
4. Run notebooks **in order**:

   ```
   stereo_calib1
   → stereo_calib2
   → stereo_calib3
   → rectify_and_undistorted
   → pointclouds_before / pointclouds_improve
   ```
5. Generated outputs:

   * Calibration parameters
   * Rectified images
   * Disparity maps
   * 3D point clouds (PLY or equivalent formats)

---

## Output Verification

* **Calibration**

  * Check reprojection error values
  * Inspect intrinsic matrices and distortion coefficients
* **Rectification**

  * Verify horizontal alignment of stereo image pairs
* **Disparity**

  * Confirm smooth and consistent depth gradients
* **Point Clouds**

  * Validate geometric structure
  * Check robustness in low-texture areas (especially improved version)

---

## Notes

* Keep stereo baseline fixed during capture.
* Avoid motion blur and extreme lighting changes.
* For accurate depth estimation, ensure sufficient disparity by maintaining appropriate camera-object distance.

---

## License

MIT License

