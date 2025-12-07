import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# =========================================================================
## ⚙️ 1. PARÂMETROS DE CALIBRAÇÃO (TUM FR1/DESK)
# Estes valores substituem o o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
# =========================================================================

# Intrínsecos do TUM fr1
FOCAL_LENGTH_X = 517.3
FOCAL_LENGTH_Y = 516.5
PRINCIPAL_POINT_X = 318.6
PRINCIPAL_POINT_Y = 255.3
WIDTH = 640 
HEIGHT = 480

# Fator de escala: O TUM armazena a profundidade como mm/5000.
DEPTH_SCALE = 5000.0 

# --- Nomes de Arquivos (Verifique sua pasta) ---
COLOR_FILE = 'cor.png'       # Arquivo RGB sincronizado
DEPTH_FILE = 'profundidade.png' # Arquivo Depth (uint16) sincronizado

if __name__ == "__main__":
    print("Iniciando leitura do TUM RGB-D Dataset...")
    
    # --- 1. Leitura de Imagens ---
    if not os.path.exists(COLOR_FILE) or not os.path.exists(DEPTH_FILE):
        print(f"ERRO: Arquivos {COLOR_FILE} ou {DEPTH_FILE} não encontrados. Verifique a pasta.")
        sys.exit()

    color_raw = o3d.io.read_image(COLOR_FILE)
    depth_raw = o3d.io.read_image(DEPTH_FILE)

    # --- 2. Criação do RGBDImage (Incluindo a escala e o tipo de imagem) ---
    print(f"Criando RGBDImage com conversão de profundidade (escala 1/{DEPTH_SCALE})...")
    
    # Para o TUM, usamos o DEPTH_SCALE para converter o uint16 (mm) para float (metros)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, 
        depth_raw, 
        depth_scale=DEPTH_SCALE,
        convert_rgb_to_intensity=False # Mantém a cor RGB
    )
    
    print(rgbd_image)

    # --- 3. Visualização 2D (Similar ao tutorial Redwood) ---
    plt.figure(figsize=(10, 5))
    
    # Imagem Colorida (RGB)
    plt.subplot(1, 2, 1)
    plt.title('TUM Color Image')
    # Nota: RGBDImage.color contém o array numpy da imagem RGB.
    plt.imshow(np.asarray(rgbd_image.color)) 
    
    # Imagem de Profundidade (Depth - em metros)
    plt.subplot(1, 2, 2)
    plt.title('TUM Depth Image (Meters)')
    # Nota: RGBDImage.depth contém o array float (metros) após a conversão.
    plt.imshow(np.asarray(rgbd_image.depth), cmap='jet')
    plt.show()

    # --- 4. Criação da Nuvem de Pontos ---
    
    # Define os Intrínsecos específicos do TUM fr1
    intrinsic_params = o3d.camera.PinholeCameraIntrinsic(
        WIDTH, HEIGHT, 
        FOCAL_LENGTH_X, FOCAL_LENGTH_Y, 
        PRINCIPAL_POINT_X, PRINCIPAL_POINT_Y
    )
    
    print("Gerando Nuvem de Pontos 3D...")
    
    # Criação da Nuvem de Pontos
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic_params
    )
    
    # Flip it, otherwise the pointcloud will be upside down (Padrão do Redwood/PrimeSense)
    # Rotação de 180 graus no eixo X para orientação correta: Y para cima, Z para profundidade.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    print(f"✅ Nuvem de pontos gerada com {len(pcd.points)} pontos.")

    # --- 5. Visualização 3D ---
    o3d.visualization.draw_geometries([pcd], window_name="Reconstrução TUM (Adaptado)")