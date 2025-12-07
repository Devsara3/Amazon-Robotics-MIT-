import numpy as np
import open3d as o3d
import imageio.v3 as iio
import os
import sys

# =========================================================================
## ⚙️ 1. TSUKUBA STEREO CALIBRATION DATA (SCALED)
# The parameters are scaled for the actual image size 384x288.
# =========================================================================

# --- Original Calibration Data (Presumed 2964x1988 resolution) ---
FOCAL_LENGTH_ORIGINAL = 3997.684
PRINCIPAL_POINT_X_ORIGINAL = 1176.728
PRINCIPAL_POINT_Y_ORIGINAL = 1011.728
WIDTH_ORIGINAL = 2964

# --- Actual Image Resolution (Read from your error output) ---
WIDTH_ACTUAL = 384
HEIGHT_ACTUAL = 288

# --- Scaling Calculation ---
SCALE_FACTOR = WIDTH_ACTUAL / WIDTH_ORIGINAL 

# --- Scaled Intrinsic Parameters ---
FOCAL_LENGTH = FOCAL_LENGTH_ORIGINAL * SCALE_FACTOR
PRINCIPAL_POINT_X = PRINCIPAL_POINT_X_ORIGINAL * SCALE_FACTOR
PRINCIPAL_POINT_Y = PRINCIPAL_POINT_Y_ORIGINAL * SCALE_FACTOR

# Extrinsic and Disparity Parameters
BASELINE_MM = 193.001       
DISPARITY_OFFSET = 131.111  
BASELINE_METERS = BASELINE_MM / 1000.0  
PGM_DISPARITY_SCALE = 1.0 

# --- File Names ---
COLOR_FILE = 'scene.ppm'
DISPARITY_FILE = 'map.pgm'

# =========================================================================
## 🔄 2. DISPARITY TO DEPTH (Z) CONVERSION (ROBUST READING)
# =========================================================================

def load_data_and_convert_to_depth(f, B_meters, offset, d_scale):
    
    if not os.path.exists(COLOR_FILE) or not os.path.exists(DISPARITY_FILE):
        print(f"ERRO: Arquivos {COLOR_FILE} ou {DISPARITY_FILE} não encontrados.")
        return None, None
    
    try:
        # 🚨 FIX: Load PPM using imageio (RGB, uint8 format)
        color_np = iio.imread(COLOR_FILE).astype(np.uint8) # Ensure correct dtype for color
        # Load PGM (Disparity)
        disparity_raw = iio.imread(DISPARITY_FILE).astype(np.float32)
    except Exception as e:
        print(f"ERRO ao ler PPM/PGM: {e}")
        return None, None

    # Verification: Ensure loaded image size matches the calibration
    if color_np.shape[0] != HEIGHT_ACTUAL or color_np.shape[1] != WIDTH_ACTUAL:
         print(f"ERRO DE DIMENSÃO: A imagem de cor tem tamanho {color_np.shape[:2]}, mas o esperado é {HEIGHT_ACTUAL}x{WIDTH_ACTUAL}.")
         return None, None
    
    # --- Disparity to Depth (Z) Calculation ---
    disparity_scaled = disparity_raw / d_scale
    disparity_corrected = disparity_scaled - offset
    valid_mask = disparity_corrected > 0.0
    depth_map = np.zeros_like(disparity_corrected, dtype=np.float32)
    depth_map[valid_mask] = (f * B_meters) / disparity_corrected[valid_mask]

    depth_map[np.isinf(depth_map)] = 0.0
    depth_map[np.isnan(depth_map)] = 0.0
    
    valid_points_count = np.count_nonzero(depth_map)
    print(f"\nDIAGNÓSTICO: {valid_points_count} pontos válidos (Z > 0) no array NumPy.")

    # 🚨 FIX: Convert NumPy arrays to Open3D Image objects
    # This step is the direct bridge from NumPy to Open3D geometry.
    color_o3d = o3d.geometry.Image(color_np)
    depth_map_o3d = o3d.geometry.Image(depth_map)
    
    # --- Final check of Open3D object sizes before exiting the function ---
    # This diagnostic helps confirm the sizes are not (0,0) before the factory method fails.
    print(f"DIAGNÓSTICO O3D: Color size: {np.asarray(color_o3d).shape}, Depth size: {np.asarray(depth_map_o3d).shape}")


    return color_o3d, depth_map_o3d

# =========================================================================
## 📸 3. GERAÇÃO DA NUVEM DE PONTOS (OPEN3D)
# =========================================================================

def create_point_cloud(color_o3d, depth_o3d):
    
    # 🚨 This is the point of failure: Open3D requires perfect size match here.
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, 
        depth_o3d, 
        convert_rgb_to_intensity=False
    )
    
    intrinsic_params = o3d.camera.PinholeCameraIntrinsic(
        WIDTH_ACTUAL, HEIGHT_ACTUAL, 
        FOCAL_LENGTH, FOCAL_LENGTH,
        PRINCIPAL_POINT_X, PRINCIPAL_POINT_Y
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic_params
    )

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd

# =========================================================================
## 🚀 4. EXECUÇÃO PRINCIPAL
# =========================================================================

if __name__ == "__main__":
    
    color_img, depth_img = load_data_and_convert_to_depth(
        FOCAL_LENGTH, 
        BASELINE_METERS,
        DISPARITY_OFFSET,
        PGM_DISPARITY_SCALE
    )
    
    if color_img is not None and depth_img is not None:
        
        point_cloud = create_point_cloud(color_img, depth_img)

        final_points = len(point_cloud.points)
        
        if final_points > 0:
            print(f"✅ Nuvem de pontos gerada com sucesso: {final_points} pontos.")
            
            output_pcd = "reconstrucao_tsukuba_final.pcd"
            o3d.io.write_point_cloud(output_pcd, point_cloud)
            print(f"Nuvem de pontos salva em: {output_pcd}")

            print("Abrindo visualização interativa da Nuvem de Pontos...")
            o3d.visualization.draw_geometries(
                [point_cloud],
                window_name="Reconstrução 3D Tsukuba (Final)"
            )
        else:
            print("\n❌ ERRO: A nuvem de pontos final está vazia.")