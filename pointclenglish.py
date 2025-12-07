import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# =========================================================================
# 📦 PROJECT SETUP AND IMPORTS
# This section imports the necessary libraries and modules.
# =========================================================================

# The 'open3d' library is essential for 3D data processing and point cloud generation.
# It handles image loading, intrinsic camera modeling, and visualization.
# Note: Matplotlib is included here mainly for 2D visualization and debugging, 
# although the final script focuses on 3D reconstruction.

# =========================================================================
# ⚙️ 1. CAMERA INTRINSICS AND CALIBRATION CONSTANTS (TUM FR1/DESK)
# These parameters define the internal geometry of the depth camera 
# for the 'freiburg1' sequence, substituting the need for a calib.txt file.
# =========================================================================

# Focal Length X (fx): The focal length measured in terms of horizontal pixels.
FOCAL_LENGTH_X = 517.3

# Focal Length Y (fy): The focal length measured in terms of vertical pixels.
FOCAL_LENGTH_Y = 516.5

# Principal Point X (cx): The X-coordinate of the optical center of the image.
PRINCIPAL_POINT_X = 318.6

# Principal Point Y (cy): The Y-coordinate of the optical center of the image.
PRINCIPAL_POINT_Y = 255.3

# Image Resolution (Width and Height in pixels).
WIDTH = 640 
HEIGHT = 480

# DEPTH_SCALE: This is the critical metric conversion factor for TUM datasets.
# The raw pixel value (uint16) represents depth in millimeters * divided by 5000*.
# The Open3D API will automatically divide the raw depth by this factor to get meters.
DEPTH_SCALE = 5000.0 

# --- File Names ---
# These are the names of the synchronized RGB and Depth files copied into the project directory.
COLOR_FILE = 'cor.png'       
DEPTH_FILE = 'profundidade.png' 

# =========================================================================
# 🔄 2. MAIN RECONSTRUCTION FUNCTION
# This function encapsulates the entire process from loading files to visualization.
# =========================================================================

def create_tum_point_cloud():
    
    # 1. File Existence Check: Ensures both required images are in the directory.
    print("1. Loading images...")
    if not os.path.exists(COLOR_FILE) or not os.path.exists(DEPTH_FILE):
        print(f"ERROR: Required files ({COLOR_FILE} or {DEPTH_FILE}) not found. Please check your directory.")
        sys.exit()
    
    # 2. Image Loading: Reading the synchronized files using Open3D's IO module.
    color_raw = o3d.io.read_image(COLOR_FILE)
    depth_raw = o3d.io.read_image(DEPTH_FILE)

    if color_raw.is_empty() or depth_raw.is_empty():
        print("ERROR: Images loaded but appear empty or corrupted.")
        sys.exit()
        
    print("Images loaded successfully.")

    # 3. Create RGBDImage Object
    # This step combines the color and depth data into a single structure for processing.
    print(f"2. Creating RGBDImage object with depth scale: 1/{DEPTH_SCALE}")
    
    # depth_scale=DEPTH_SCALE: Tells Open3D to divide the raw depth pixel value (uint16) by 5000.0, 
    # resulting in the final depth (Z) value in meters.
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, 
        depth_raw, 
        depth_scale=DEPTH_SCALE, 
        convert_rgb_to_intensity=False # Keeps the original RGB color format.
    )
    
    # 4. Define Pinhole Camera Intrinsics
    # This object contains the K matrix (focal lengths and principal points) 
    # necessary for projecting 2D pixels back to 3D coordinates.
    intrinsic_params = o3d.camera.PinholeCameraIntrinsic(
        WIDTH, HEIGHT, 
        FOCAL_LENGTH_X, FOCAL_LENGTH_Y, 
        PRINCIPAL_POINT_X, PRINCIPAL_POINT_Y
    )
    
    # 5. Generate Point Cloud
    # This is the core step: transforming the 2D RGBD image data into a 3D point cloud.
    # The intrinsic parameters define the projection geometry.
    print("3. Generating 3D Point Cloud...")
    
    # Note: We use the common create_from_rgbd_image method for TUM datasets 
    # since the input data types are well-supported (uint8 color, uint16 depth).
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic_params
    )
    
    # 6. Apply Orientation Correction (Flipping)
    # The standard coordinate system for RGB-D sensors (Z forward, Y down) often 
    # results in an upside-down point cloud in Open3D's visualization.
    # This transformation applies a 180-degree flip around the X-axis: 
    # (Y -> -Y, Z -> -Z), making Y point upwards. 
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    final_points = len(pcd.points)
    print(f"✅ Point Cloud generated with {final_points} points.")

    # 7. Visualization and Saving
    if final_points > 0:
        # Save the resulting Point Cloud to a standard format (PCD)
        o3d.io.write_point_cloud("reconstrucao_tum_desk.pcd", pcd)
        print("File saved as 'reconstrucao_tum_desk.pcd'")
        
        # Display the 3D model in an interactive viewer
        o3d.visualization.draw_geometries([pcd], window_name="TUM RGB-D Reconstruction")
    else:
        print("❌ Point cloud generation failed. Final point count is zero.")

# =========================================================================
# 🚀 EXECUTION
# =========================================================================

if __name__ == "__main__":
    # Check for the matplotlib module (required for the visualization tutorial structure)
    try:
        # Note: The original tutorial code included Matplotlib visualization for 2D images.
        # This section ensures that the 3D reconstruction runs even if Matplotlib is missing.
        if 'matplotlib' not in sys.modules:
             # Running the main function directly without 2D plots for simplicity/robustness
            create_tum_point_cloud()
        else:
            # If Matplotlib is installed, you could add the 2D visualization here if desired.
            create_tum_point_cloud()
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit()