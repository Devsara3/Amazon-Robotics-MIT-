import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import json

# =========================================================================
## ⚙️ 1. PARÂMETROS E ARQUIVOS
# =========================================================================

COLOR_FILE = 'cor.jpg'
DISPARITY_FILE = 'gray.jpg'
PARAMS_JSON = 'map.json'
STEREO_PARAMS = 'stereo_params.npz'

# =========================================================================
## 🔧 2. FILTROS DE REDUÇÃO DE RUÍDO
# =========================================================================

def filter_depth_map(depth, method='bilateral', kernel_size=5):
    """
    Aplica filtros para reduzir ruído no mapa de profundidade.
    
    Args:
        depth: Mapa de profundidade (float32)
        method: 'bilateral', 'median', 'gaussian', 'morphology', 'combined'
        kernel_size: Tamanho do kernel (ímpar)
    
    Returns:
        depth_filtered: Mapa filtrado
    """
    
    print(f"\n🔧 Aplicando filtro: {method}")
    
    # Criar máscara de pixels válidos
    valid_mask = depth > 0
    
    if method == 'bilateral':
        # Filtro bilateral: suaviza mantendo bordas
        depth_filtered = cv2.bilateralFilter(
            depth.astype(np.float32), 
            d=kernel_size,
            sigmaColor=50,
            sigmaSpace=50
        )
        
    elif method == 'median':
        # Filtro de mediana: remove ruído sal e pimenta
        depth_uint16 = (depth).astype(np.uint16)
        depth_filtered = cv2.medianBlur(depth_uint16, kernel_size)
        depth_filtered = depth_filtered.astype(np.float32)
        
    elif method == 'gaussian':
        # Filtro gaussiano: suavização simples
        depth_filtered = cv2.GaussianBlur(
            depth.astype(np.float32),
            (kernel_size, kernel_size),
            0
        )
        
    elif method == 'morphology':
        # Morphological closing: fecha pequenos buracos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        depth_uint16 = (depth).astype(np.uint16)
        depth_filtered = cv2.morphologyEx(depth_uint16, cv2.MORPH_CLOSE, kernel)
        depth_filtered = depth_filtered.astype(np.float32)
        
    elif method == 'combined':
        # Combinação de filtros
        print("   Passo 1: Mediana (remove ruído)")
        depth_uint16 = (depth).astype(np.uint16)
        depth_filtered = cv2.medianBlur(depth_uint16, 5)
        
        print("   Passo 2: Bilateral (suaviza mantendo bordas)")
        depth_filtered = cv2.bilateralFilter(
            depth_filtered.astype(np.float32),
            d=9,
            sigmaColor=75,
            sigmaSpace=75
        )
        
        print("   Passo 3: Morphology (fecha buracos)")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        depth_uint16 = depth_filtered.astype(np.uint16)
        depth_filtered = cv2.morphologyEx(depth_uint16, cv2.MORPH_CLOSE, kernel)
        depth_filtered = depth_filtered.astype(np.float32)
        
    else:
        depth_filtered = depth
    
    # Restaurar zeros (pixels inválidos)
    depth_filtered[~valid_mask] = 0
    
    # Estatísticas
    valid_filtered = depth_filtered[valid_mask]
    if len(valid_filtered) > 0:
        print(f"   ✓ Após filtro: min={valid_filtered.min():.2f}, max={valid_filtered.max():.2f}")
    
    return depth_filtered


def fill_depth_holes(depth, max_hole_size=10):
    """Preenche pequenos buracos no mapa de profundidade."""
    
    print(f"\n🔧 Preenchendo buracos (max_size={max_hole_size})")
    
    # Criar máscara de buracos (zeros no meio de valores válidos)
    mask_holes = (depth == 0).astype(np.uint8)
    
    # Usar inpainting para preencher buracos pequenos
    depth_filled = cv2.inpaint(
        depth.astype(np.float32),
        mask_holes,
        inpaintRadius=max_hole_size,
        flags=cv2.INPAINT_TELEA
    )
    
    # Contar buracos preenchidos
    holes_filled = np.sum((depth == 0) & (depth_filled > 0))
    print(f"   ✓ Pixels preenchidos: {holes_filled:,}")
    
    return depth_filled


def remove_outliers_depth(depth, threshold=3.0):
    """Remove outliers estatísticos do mapa de profundidade."""
    
    print(f"\n🔧 Removendo outliers (threshold={threshold} std)")
    
    valid_mask = depth > 0
    valid_depths = depth[valid_mask]
    
    if len(valid_depths) == 0:
        return depth
    
    mean = np.mean(valid_depths)
    std = np.std(valid_depths)
    
    # Marcar outliers
    outlier_mask = (depth > 0) & ((depth < mean - threshold * std) | (depth > mean + threshold * std))
    
    depth_clean = depth.copy()
    depth_clean[outlier_mask] = 0
    
    print(f"   ✓ Outliers removidos: {np.sum(outlier_mask):,}")
    
    return depth_clean


# =========================================================================
## 📊 3. FUNÇÕES DE CARREGAMENTO
# =========================================================================

def load_disparity_map(gray_path, json_path):
    """Carrega mapa de disparidade e reverte normalização."""
    
    print(f"📂 Carregando disparidade: {gray_path}")
    disp_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    
    if disp_img is None:
        print(f"❌ ERRO: Não foi possível carregar {gray_path}")
        return None
    
    print(f"   Shape: {disp_img.shape}, Range: {disp_img.min()}-{disp_img.max()}")
    
    # Tentar obter max_disp do JSON
    max_disp = 128
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                params = json.load(f)
            
            if 'numDisparities' in params and params['numDisparities'] > 0:
                max_disp = params['numDisparities'] * 16
                print(f"   ✓ numDisparities do JSON: {params['numDisparities']} → max_disp = {max_disp}")
        except:
            pass
    
    if max_disp == 0:
        print("\n⚠️  numDisparities não encontrado ou zero no JSON")
        user_input = input(f"   Digite max_disp (Enter para {128}): ").strip()
        max_disp = float(user_input) if user_input else 128
    
    # Reverter normalização
    disparity = (disp_img.astype(np.float32) / 255.0) * max_disp
    disparity[disp_img == 0] = 0
    
    print(f"   ✓ Revertida: Range {disparity.min():.2f}-{disparity.max():.2f}")
    
    return disparity


def disparity_to_depth(disparity, focal_length, baseline):
    """Converte disparidade para profundidade."""
    
    print(f"\n🔄 Convertendo disparidade → profundidade")
    print(f"   Focal: {focal_length:.2f} pixels")
    print(f"   Baseline: {baseline:.2f} mm")
    
    depth = np.zeros_like(disparity, dtype=np.float32)
    
    valid_mask = disparity > 0
    depth[valid_mask] = (focal_length * baseline) / disparity[valid_mask]
    
    valid_depth = depth[valid_mask]
    if len(valid_depth) > 0:
        print(f"   ✓ Profundidade: min={valid_depth.min():.2f} mm, max={valid_depth.max():.2f} mm")
        print(f"   ✓ Pixels válidos: {np.sum(valid_mask):,} ({100*np.sum(valid_mask)/depth.size:.1f}%)")
    
    return depth


def load_camera_params(stereo_params_path, img_width, img_height):
    """Carrega parâmetros da câmera."""
    
    focal_length = None
    baseline = None
    cx = img_width / 2
    cy = img_height / 2
    
    if os.path.exists(stereo_params_path):
        print(f"\n📷 Carregando calibração: {stereo_params_path}")
        try:
            data = np.load(stereo_params_path)
            
            if 'mtx_L' in data:
                focal_length = float(data['mtx_L'][0, 0])
                cx = float(data['mtx_L'][0, 2])
                cy = float(data['mtx_L'][1, 2])
                print(f"   ✓ Focal: {focal_length:.2f} pixels")
                print(f"   ✓ Principal point: ({cx:.2f}, {cy:.2f})")
            
            if 'T' in data:
                baseline = float(np.linalg.norm(data['T']))
                print(f"   ✓ Baseline: {baseline:.2f} mm")
                
        except Exception as e:
            print(f"   ⚠️ Erro: {e}")
    
    if focal_length is None:
        focal_length = img_width
        print(f"   ⚠️ Usando focal estimado: {focal_length} pixels")
    
    if baseline is None:
        baseline = 600.0
        print(f"   ⚠️ Usando baseline estimado: {baseline} mm")
    
    return focal_length, baseline, cx, cy


# =========================================================================
## 🎯 4. SCRIPT PRINCIPAL
# =========================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("🚀 RECONSTRUÇÃO 3D - COM REDUÇÃO DE RUÍDO")
    print("="*70)
    
    # --- Verificar arquivos ---
    print("\n📁 Verificando arquivos...")
    
    if not os.path.exists(COLOR_FILE):
        print(f"❌ ERRO: {COLOR_FILE} não encontrado")
        sys.exit(1)
    
    if not os.path.exists(DISPARITY_FILE):
        print(f"❌ ERRO: {DISPARITY_FILE} não encontrado")
        sys.exit(1)
    
    print(f"   ✓ {COLOR_FILE}")
    print(f"   ✓ {DISPARITY_FILE}")
    
    # --- 1. Carregar imagem RGB ---
    print("\n" + "="*70)
    print("1️⃣  CARREGANDO IMAGEM RGB")
    print("="*70)
    
    color_bgr = cv2.imread(COLOR_FILE)
    if color_bgr is None:
        print(f"❌ ERRO: Não foi possível carregar {COLOR_FILE}")
        sys.exit(1)
    
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    height, width = color_rgb.shape[:2]
    print(f"✓ Imagem carregada: {width}×{height} pixels")
    
    # --- 2. Carregar disparidade ---
    print("\n" + "="*70)
    print("2️⃣  CARREGANDO DISPARIDADE")
    print("="*70)
    
    disparity = load_disparity_map(DISPARITY_FILE, PARAMS_JSON)
    if disparity is None:
        sys.exit(1)
    
    if disparity.shape != (height, width):
        print(f"\n⚠️  Redimensionando disparidade de {disparity.shape} para {(height, width)}")
        disparity = cv2.resize(disparity, (width, height))
    
    # --- 3. Parâmetros da câmera ---
    print("\n" + "="*70)
    print("3️⃣  PARÂMETROS DA CÂMERA")
    print("="*70)
    
    focal_length, baseline, cx, cy = load_camera_params(STEREO_PARAMS, width, height)
    
    # --- 4. Converter para profundidade ---
    print("\n" + "="*70)
    print("4️⃣  CONVERSÃO DISPARIDADE → PROFUNDIDADE")
    print("="*70)
    
    depth_raw = disparity_to_depth(disparity, focal_length, baseline)
    
    # --- 5. APLICAR FILTROS ---
    print("\n" + "="*70)
    print("5️⃣  REDUÇÃO DE RUÍDO")
    print("="*70)
    
    print("\nEscolha o nível de filtragem:")
    print("  [1] Leve    - Bilateral apenas (rápido)")
    print("  [2] Médio   - Bilateral + Mediana (recomendado)")
    print("  [3] Pesado  - Combinado + Preenchimento (mais lento)")
    print("  [4] Nenhum  - Sem filtros")
    
    choice = input("\nEscolha (1/2/3/4): ").strip() or "2"
    
    if choice == "1":
        depth_filtered = filter_depth_map(depth_raw, method='bilateral', kernel_size=9)
    elif choice == "2":
        depth_temp = filter_depth_map(depth_raw, method='median', kernel_size=5)
        depth_filtered = filter_depth_map(depth_temp, method='bilateral', kernel_size=9)
    elif choice == "3":
        depth_filtered = filter_depth_map(depth_raw, method='combined')
        depth_filtered = fill_depth_holes(depth_filtered, max_hole_size=5)
        depth_filtered = remove_outliers_depth(depth_filtered, threshold=2.5)
    else:
        depth_filtered = depth_raw
        print("   ⚠️ Sem filtros aplicados")
    
    # --- 6. Visualização 2D (Antes vs Depois) ---
    print("\n" + "="*70)
    print("6️⃣  VISUALIZAÇÃO 2D")
    print("="*70)
    
    plt.figure(figsize=(15, 10))
    
    # RGB
    plt.subplot(2, 3, 1)
    plt.title('Imagem RGB')
    plt.imshow(color_rgb)
    plt.axis('off')
    
    # Disparidade
    plt.subplot(2, 3, 2)
    plt.title('Disparidade (pixels)')
    plt.imshow(disparity, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    
    # Profundidade RAW
    plt.subplot(2, 3, 3)
    plt.title('Profundidade RAW (mm)')
    depth_display_raw = np.copy(depth_raw)
    depth_display_raw[depth_display_raw == 0] = np.nan
    plt.imshow(depth_display_raw, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    
    # Profundidade FILTRADA
    plt.subplot(2, 3, 6)
    plt.title('Profundidade FILTRADA (mm)')
    depth_display_filtered = np.copy(depth_filtered)
    depth_display_filtered[depth_display_filtered == 0] = np.nan
    im = plt.imshow(depth_display_filtered, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    
    # Comparação (diferença)
    plt.subplot(2, 3, 5)
    plt.title('Diferença (Filtrada - Raw)')
    diff = np.abs(depth_filtered - depth_raw)
    diff[depth_raw == 0] = 0
    plt.imshow(diff, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # --- 7. Criar RGBDImage ---
    print("\n" + "="*70)
    print("7️⃣  CRIANDO RGBD IMAGE")
    print("="*70)
    
    color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_filtered.astype(np.float32))
    
    depth_trunc = np.nanpercentile(depth_filtered[depth_filtered > 0], 95) * 1.2
    
    print(f"   depth_trunc = {depth_trunc:.2f} mm")
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )
    
    print(f"✓ RGBDImage criado")
    
    # --- 8. Parâmetros intrínsecos ---
    print("\n" + "="*70)
    print("8️⃣  PARÂMETROS INTRÍNSECOS")
    print("="*70)
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        focal_length, focal_length,
        cx, cy
    )
    
    print(f"   Width: {width}, Height: {height}")
    print(f"   Focal: {focal_length:.2f}")
    print(f"   Principal: ({cx:.2f}, {cy:.2f})")
    
    # --- 9. Criar nuvem de pontos ---
    print("\n" + "="*70)
    print("9️⃣  GERANDO NUVEM DE PONTOS 3D")
    print("="*70)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )
    
    pcd.transform([[1, 0, 0, 0], 
                   [0, -1, 0, 0], 
                   [0, 0, -1, 0], 
                   [0, 0, 0, 1]])
    
    print(f"✓ Nuvem de pontos inicial: {len(pcd.points):,} pontos")
    
    # Remover outliers estatísticos da nuvem 3D
    print("\n🔧 Removendo outliers da nuvem 3D...")
    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    print(f"   ✓ Após remoção: {len(pcd_clean.points):,} pontos")
    print(f"   ✓ Removidos: {len(pcd.points) - len(pcd_clean.points):,} pontos")
    
    # Estatísticas
    points = np.asarray(pcd_clean.points)
    if len(points) > 0:
        print(f"\n📊 Estatísticas finais:")
        print(f"   X: {points[:, 0].min():.2f} a {points[:, 0].max():.2f} mm")
        print(f"   Y: {points[:, 1].min():.2f} a {points[:, 1].max():.2f} mm")
        print(f"   Z: {points[:, 2].min():.2f} a {points[:, 2].max():.2f} mm")
    
    # --- 10. Salvar ---
    output_ply = "reconstruction_3d_filtered.ply"
    
    print(f"\n💾 Salvando: {output_ply}")
    o3d.io.write_point_cloud(output_ply, pcd_clean)
    print(f"✓ Salvo!")
    
    # --- 11. Visualização 3D ---
    print("\n" + "="*70)
    print("🔟 VISUALIZAÇÃO 3D")
    print("="*70)
    
    o3d.visualization.draw_geometries(
        [pcd_clean],
        window_name="Reconstrução 3D - Filtrada",
        width=1280,
        height=720
    )
    
    print("\n" + "="*70)
    print("✅ CONCLUÍDO!")
    print("="*70)
    print(f"\nArquivo: {output_ply}")
    print(f"Pontos: {len(pcd_clean.points):,}")
    print("="*70)