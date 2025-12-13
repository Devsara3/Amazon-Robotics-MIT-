import numpy as np
import cv2
import os

# --- PARÂMETROS (AJUSTE CONFORME NECESSÁRIO) ---
CHECKERBOARD = (7, 7)  # Tente também (6,9), (9,6), etc se não funcionar
SQUARE_SIZE = 25.0
IMAGE_FOLDER = '.'

# Flags adicionais para melhorar detecção
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = []
imgpoints_L = []
imgpoints_R = []

print("Procurando pares de imagens...")
num_pairs = 14

image_files = []
for i in range(1, num_pairs + 1):
    fname_L = os.path.join(IMAGE_FOLDER, f'left{i}.jpg')
    fname_R = os.path.join(IMAGE_FOLDER, f'right{i}.jpg')
    
    if os.path.exists(fname_L) and os.path.exists(fname_R):
        image_files.append((fname_L, fname_R))

if not image_files:
    print("ERRO: Nenhuma imagem encontrada.")
    exit()

print(f"Total de {len(image_files)} pares encontrados.\n")

# --- PRÉ-PROCESSAMENTO E DETECÇÃO MELHORADA ---

for idx, (fname_L, fname_R) in enumerate(image_files):
    print(f"Processando par {idx+1}/{len(image_files)}...", end=" ")
    
    img_L = cv2.imread(fname_L)
    img_R = cv2.imread(fname_R)
    
    if img_L is None or img_R is None:
        print("❌ Erro ao carregar imagens")
        continue
    
    # Redimensionar se imagens muito grandes (> 2000px)
    scale = 1.0
    if max(img_L.shape[:2]) > 2000:
        scale = 2000 / max(img_L.shape[:2])
        img_L = cv2.resize(img_L, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        img_R = cv2.resize(img_R, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
    
    # TÉCNICA 1: Tentar com equalização de histograma
    gray_L_eq = cv2.equalizeHist(gray_L)
    gray_R_eq = cv2.equalizeHist(gray_R)
    
    # Tenta encontrar cantos (primeiro na imagem equalizada)
    ret_L, corners_L = cv2.findChessboardCorners(gray_L_eq, CHECKERBOARD, flags=flags)
    ret_R, corners_R = cv2.findChessboardCorners(gray_R_eq, CHECKERBOARD, flags=flags)
    
    # Se não funcionou, tenta na imagem original
    if not ret_L:
        ret_L, corners_L = cv2.findChessboardCorners(gray_L, CHECKERBOARD, flags=flags)
    if not ret_R:
        ret_R, corners_R = cv2.findChessboardCorners(gray_R, CHECKERBOARD, flags=flags)
    
    if ret_L and ret_R:
        # Refinar cantos na imagem original
        corners_L = cv2.cornerSubPix(gray_L, corners_L, (11, 11), (-1, -1), criteria)
        corners_R = cv2.cornerSubPix(gray_R, corners_R, (11, 11), (-1, -1), criteria)
        
        objpoints.append(objp)
        imgpoints_L.append(corners_L)
        imgpoints_R.append(corners_R)
        
        print("✅ Detectado")
        
        # VISUALIZAÇÃO (descomente para debug)
        # img_L_vis = cv2.drawChessboardCorners(img_L.copy(), CHECKERBOARD, corners_L, ret_L)
        # img_R_vis = cv2.drawChessboardCorners(img_R.copy(), CHECKERBOARD, corners_R, ret_R)
        # vis = np.hstack((img_L_vis, img_R_vis))
        # cv2.imshow('Detecção', cv2.resize(vis, None, fx=0.5, fy=0.5))
        # cv2.waitKey(500)
    else:
        print(f"❌ Falhou (L:{ret_L}, R:{ret_R})")
        
        # SALVAR IMAGEM PROBLEMÁTICA PARA ANÁLISE
        if idx < 3:  # Salva os 3 primeiros problemas
            cv2.imwrite(f'debug_left{idx+1}.jpg', gray_L_eq)
            cv2.imwrite(f'debug_right{idx+1}.jpg', gray_R_eq)
            print(f"   → Imagens de debug salvas: debug_left{idx+1}.jpg / debug_right{idx+1}.jpg")

cv2.destroyAllWindows()

print(f"\n📊 Resumo: {len(objpoints)} pares válidos de {len(image_files)} totais")

if len(objpoints) < 8:
    print(f"\n⚠️  ERRO: Apenas {len(objpoints)} pares válidos. Mínimo recomendado: 10-15")
    print("\n🔍 Checklist de Troubleshooting:")
    print("   1. Verifique o tamanho do tabuleiro (tente 6x9 ou 9x6)")
    print("   2. Confira as imagens de debug salvas")
    print("   3. Certifique-se que o tabuleiro está bem iluminado")
    print("   4. O tabuleiro deve estar completamente visível")
    print("   5. Evite reflexos e sombras fortes")
    exit()

# Resto do código de calibração...
h, w = gray_L.shape[:2]

print("\n🔧 Iniciando Calibração Monocular...")
ret_L, mtx_L, dist_L, _, _ = cv2.calibrateCamera(objpoints, imgpoints_L, (w, h), None, None)
ret_R, mtx_R, dist_R, _, _ = cv2.calibrateCamera(objpoints, imgpoints_R, (w, h), None, None)

print("🔧 Iniciando Calibração Estéreo...")
retStereo, mtx_L, dist_L, mtx_R, dist_R, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_L, imgpoints_R,
    mtx_L, dist_L, mtx_R, dist_R,
    (w, h), criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)

print("\n✅ CALIBRAÇÃO CONCLUÍDA")
print(f"Erro RMS Estéreo: {retStereo:.4f}")
if retStereo > 1.0:
    print("⚠️  Aviso: Erro alto (> 1.0). Considere mais imagens ou melhor qualidade.")

print(f"\nBaseline (distância entre câmeras): {np.linalg.norm(T):.2f} mm")

np.savez('stereo_params.npz', 
         retStereo=retStereo, mtx_L=mtx_L, dist_L=dist_L, 
         mtx_R=mtx_R, dist_R=dist_R, R=R, T=T, E=E, F=F)
print("\n💾 Parâmetros salvos em 'stereo_params.npz'")