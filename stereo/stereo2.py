import numpy as np
import cv2
import os
import json
from datetime import datetime

# --- CONFIGURAÇÕES ---
IMAGE_LEFT_FILE = 'left.jpg'
IMAGE_RIGHT_FILE = 'right.jpg'
PARAMS_FOLDER = 'saved_configs'

# Variáveis globais
rect_L = None
rect_R = None
stereo = None
initialized = False
last_update_time = 0
current_disp_color = None
current_disp_gray = None

def preprocess_image(img):
    """Pré-processamento para melhorar qualidade."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img)
    img_filtered = cv2.bilateralFilter(img_eq, 5, 75, 75)
    return img_filtered


def post_process_disparity(disp, filter_strength='medium'):
    """Pós-processamento para reduzir ruído sem WLS."""
    disp_clean = disp.copy()
    disp_clean[disp_clean < 0] = 0
    
    if filter_strength == 'light':
        kernel_size = 3
    elif filter_strength == 'medium':
        kernel_size = 5
    else:  # heavy
        kernel_size = 7
    
    disp_median = cv2.medianBlur(disp_clean.astype(np.uint8), kernel_size)
    disp_clean = disp_median.astype(np.float32)
    
    disp_clean = cv2.bilateralFilter(
        disp_clean.astype(np.uint8), 9, 75, 75
    ).astype(np.float32)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    disp_clean = cv2.morphologyEx(
        disp_clean.astype(np.uint8), cv2.MORPH_CLOSE, kernel
    ).astype(np.float32)
    
    return disp_clean


def load_images():
    """Carrega e pré-processa as imagens."""
    global rect_L, rect_R
    
    print("\n" + "="*60)
    print("Carregando imagens...")
    print("="*60)
    
    imgL = cv2.imread(IMAGE_LEFT_FILE, 0) 
    imgR = cv2.imread(IMAGE_RIGHT_FILE, 0)
    
    if imgL is None or imgR is None:
        print(f"❌ ERRO: Não foi possível carregar as imagens!")
        return False
    
    print(f"\n✓ Imagens originais: {imgL.shape}")
    
    print("Aplicando pré-processamento (CLAHE + bilateral)...")
    rect_L = preprocess_image(imgL)
    rect_R = preprocess_image(imgR)
    
    cv2.imwrite('preprocessed_left.jpg', rect_L)
    cv2.imwrite('preprocessed_right.jpg', rect_R)
    print("✓ Pré-processamento concluído")
    
    return True


def get_current_params():
    """Retorna os parâmetros atuais dos trackbars."""
    params = {
        'blockSize': cv2.getTrackbarPos('blockSize', 'Controls'),
        'numDisparities': cv2.getTrackbarPos('numDisparities', 'Controls'),
        'uniquenessRatio': cv2.getTrackbarPos('uniquenessRatio', 'Controls'),
        'speckleWindow': cv2.getTrackbarPos('speckleWindow', 'Controls'),
        'speckleRange': cv2.getTrackbarPos('speckleRange', 'Controls'),
        'disp12MaxDiff': cv2.getTrackbarPos('disp12MaxDiff', 'Controls'),
        'preFilterCap': cv2.getTrackbarPos('preFilterCap', 'Controls'),
        'postFilter': cv2.getTrackbarPos('postFilter', 'Controls'),
        'medianFilter': cv2.getTrackbarPos('medianFilter', 'Controls')
    }
    return params


def save_configuration(name=None):
    """Salva o mapa de disparidade e os parâmetros."""
    global current_disp_color, current_disp_gray
    
    if current_disp_color is None:
        print("❌ Nenhuma disparidade para salvar!")
        return
    
    # Criar pasta se não existir
    os.makedirs(PARAMS_FOLDER, exist_ok=True)
    
    # Gerar nome automático ou usar fornecido
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"disparity_{timestamp}"
    
    base_path = os.path.join(PARAMS_FOLDER, name)
    
    # Salvar imagens
    cv2.imwrite(f"{base_path}_color.jpg", current_disp_color)
    cv2.imwrite(f"{base_path}_gray.jpg", current_disp_gray)
    
    # Salvar parâmetros
    params = get_current_params()
    params['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params['left_image'] = IMAGE_LEFT_FILE
    params['right_image'] = IMAGE_RIGHT_FILE
    
    with open(f"{base_path}_params.json", 'w') as f:
        json.dump(params, f, indent=4)
    
    # Salvar parâmetros em texto legível também
    with open(f"{base_path}_params.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("PARÂMETROS DE DISPARIDADE\n")
        f.write("="*60 + "\n\n")
        f.write(f"Data/Hora: {params['timestamp']}\n")
        f.write(f"Imagem Left: {params['left_image']}\n")
        f.write(f"Imagem Right: {params['right_image']}\n\n")
        f.write("PARÂMETROS SGBM:\n")
        f.write("-"*60 + "\n")
        f.write(f"  blockSize:        {params['blockSize']}\n")
        f.write(f"  numDisparities:   {params['numDisparities']} (x16 = {params['numDisparities']*16})\n")
        f.write(f"  uniquenessRatio:  {params['uniquenessRatio']}\n")
        f.write(f"  speckleWindow:    {params['speckleWindow']}\n")
        f.write(f"  speckleRange:     {params['speckleRange']}\n")
        f.write(f"  disp12MaxDiff:    {params['disp12MaxDiff']}\n")
        f.write(f"  preFilterCap:     {params['preFilterCap']}\n\n")
        f.write("PÓS-PROCESSAMENTO:\n")
        f.write("-"*60 + "\n")
        f.write(f"  postFilter:       {params['postFilter']} ({['None', 'Light', 'Medium', 'Heavy'][params['postFilter']]})\n")
        f.write(f"  medianFilter:     {params['medianFilter']}\n")
        f.write("="*60 + "\n")
    
    print(f"\n✓✓✓ CONFIGURAÇÃO SALVA ✓✓✓")
    print(f"Pasta: {PARAMS_FOLDER}/")
    print(f"Arquivos criados:")
    print(f"  • {name}_color.jpg     (mapa colorido)")
    print(f"  • {name}_gray.jpg      (mapa grayscale)")
    print(f"  • {name}_params.json   (parâmetros JSON)")
    print(f"  • {name}_params.txt    (parâmetros legíveis)")
    print()


def list_saved_configurations():
    """Lista todas as configurações salvas."""
    if not os.path.exists(PARAMS_FOLDER):
        print("❌ Nenhuma configuração salva ainda.")
        return []
    
    json_files = [f for f in os.listdir(PARAMS_FOLDER) if f.endswith('_params.json')]
    configs = []
    
    print("\n" + "="*60)
    print("CONFIGURAÇÕES SALVAS:")
    print("="*60)
    
    for i, json_file in enumerate(sorted(json_files), 1):
        config_name = json_file.replace('_params.json', '')
        json_path = os.path.join(PARAMS_FOLDER, json_file)
        
        try:
            with open(json_path, 'r') as f:
                params = json.load(f)
            
            print(f"\n[{i}] {config_name}")
            print(f"    Data: {params.get('timestamp', 'N/A')}")
            print(f"    blockSize={params['blockSize']}, numDisp={params['numDisparities']*16}")
            
            configs.append((config_name, params))
        except:
            pass
    
    if not configs:
        print("❌ Nenhuma configuração válida encontrada.")
    
    print("="*60 + "\n")
    return configs


def load_configuration(config_name):
    """Carrega uma configuração salva."""
    json_path = os.path.join(PARAMS_FOLDER, f"{config_name}_params.json")
    
    if not os.path.exists(json_path):
        print(f"❌ Configuração '{config_name}' não encontrada!")
        return False
    
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        
        print(f"\n✓ Carregando configuração: {config_name}")
        
        # Aplicar aos trackbars
        cv2.setTrackbarPos('blockSize', 'Controls', params['blockSize'])
        cv2.setTrackbarPos('numDisparities', 'Controls', params['numDisparities'])
        cv2.setTrackbarPos('uniquenessRatio', 'Controls', params['uniquenessRatio'])
        cv2.setTrackbarPos('speckleWindow', 'Controls', params['speckleWindow'])
        cv2.setTrackbarPos('speckleRange', 'Controls', params['speckleRange'])
        cv2.setTrackbarPos('disp12MaxDiff', 'Controls', params['disp12MaxDiff'])
        cv2.setTrackbarPos('preFilterCap', 'Controls', params['preFilterCap'])
        cv2.setTrackbarPos('postFilter', 'Controls', params['postFilter'])
        cv2.setTrackbarPos('medianFilter', 'Controls', params['medianFilter'])
        
        print("✓ Parâmetros aplicados. Recalculando disparidade...")
        compute_disparity()
        
        return True
    
    except Exception as e:
        print(f"❌ Erro ao carregar: {e}")
        return False


def compute_disparity():
    """Calcula disparidade com pós-processamento."""
    global stereo, rect_L, rect_R, last_update_time
    global current_disp_color, current_disp_gray
    
    import time
    current_time = time.time()
    
    if current_time - last_update_time < 0.1:
        return
    
    last_update_time = current_time
    
    try:
        # Ler parâmetros
        params = get_current_params()
        
        window_size = max(3, params['blockSize'])
        if window_size % 2 == 0:
            window_size += 1
        
        num_disp = params['numDisparities'] * 16
        if num_disp < 16:
            num_disp = 16
        
        # Configurar SGBM
        P1 = 8 * 3 * window_size**2
        P2 = 32 * 3 * window_size**2
        
        stereo.setBlockSize(window_size)
        stereo.setNumDisparities(num_disp)
        stereo.setUniquenessRatio(params['uniquenessRatio'])
        stereo.setSpeckleWindowSize(params['speckleWindow'])
        stereo.setSpeckleRange(params['speckleRange'])
        stereo.setDisp12MaxDiff(params['disp12MaxDiff'])
        stereo.setPreFilterCap(params['preFilterCap'])
        stereo.setP1(P1)
        stereo.setP2(P2)
        
        print(f"\r[CALC] block={window_size}, numDisp={num_disp}, filter={params['postFilter']}  ", 
              end='', flush=True)
        
        # Calcular disparidade
        disp_raw = stereo.compute(rect_L, rect_R).astype(np.float32) / 16.0
        
        # Pós-processar
        filter_modes = ['None', 'light', 'medium', 'heavy']
        if params['postFilter'] == 0:
            disp = disp_raw
        else:
            disp = post_process_disparity(disp_raw, filter_modes[params['postFilter']])
        
        # Filtro de mediana adicional
        if params['medianFilter'] > 0:
            median_k = params['medianFilter'] * 2 + 1
            disp = cv2.medianBlur(disp.astype(np.uint8), median_k).astype(np.float32)
        
        # Normalizar
        min_disp = 0
        valid_mask = disp > min_disp
        valid_pixels = np.sum(valid_mask)
        
        if valid_pixels > 0:
            valid_disp = disp[valid_mask]
            min_val = valid_disp.min()
            max_val = valid_disp.max()
            
            if max_val > min_val:
                disp_norm = 255 * (disp - min_val) / (max_val - min_val)
                disp_norm[~valid_mask] = 0
            else:
                disp_norm = np.zeros_like(disp)
        else:
            disp_norm = np.zeros_like(disp)
        
        disp_norm = np.uint8(disp_norm)
        
        # Criar visualizações
        current_disp_gray = disp_norm.copy()
        current_disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        
        # Comparação
        disp_raw_norm = np.uint8(255 * (disp_raw - disp_raw.min()) / 
                                  (disp_raw.max() - disp_raw.min() + 1e-6))
        disp_raw_color = cv2.applyColorMap(disp_raw_norm, cv2.COLORMAP_JET)
        comparison = np.hstack((disp_raw_color, current_disp_color))
        
        # Labels
        cv2.putText(comparison, "RAW", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "FILTERED", (disp_raw_color.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Info
        info_text = [
            f"blockSize: {window_size}",
            f"numDisp: {num_disp}",
            f"Valid: {100*valid_pixels/disp.size:.1f}%",
            f"Filter: {['None', 'Light', 'Medium', 'Heavy'][params['postFilter']]}"
        ]
        
        y_pos = 60
        for text in info_text:
            cv2.putText(current_disp_color, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(current_disp_color, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_pos += 20
        
        # Mostrar
        cv2.imshow('Comparison (Raw vs Filtered)', comparison)
        cv2.imshow('Disparity', current_disp_color)
        cv2.imshow('Disparity (Gray)', current_disp_gray)
        cv2.imshow('Left', rect_L)
        cv2.imshow('Right', rect_R)
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


def on_trackbar(val):
    if initialized:
        compute_disparity()


if __name__ == "__main__":
    
    print("="*60)
    print("VISÃO ESTÉREO - COM SALVAMENTO DE CONFIGURAÇÕES")
    print("="*60)
    
    if not load_images():
        exit()
    
    print("\n✓ Imagens carregadas")
    print("Inicializando StereoSGBM...")
    
    # Parâmetros iniciais
    initial_params = {
        'blockSize': 9,
        'numDisparities': 8,
        'uniquenessRatio': 15,
        'speckleWindow': 150,
        'speckleRange': 2,
        'disp12MaxDiff': 1,
        'preFilterCap': 31,
        'postFilter': 2,
        'medianFilter': 2
    }
    
    # Criar SGBM
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=initial_params['numDisparities'] * 16,
        blockSize=initial_params['blockSize'],
        uniquenessRatio=initial_params['uniquenessRatio'],
        speckleRange=initial_params['speckleRange'],
        speckleWindowSize=initial_params['speckleWindow'],
        disp12MaxDiff=initial_params['disp12MaxDiff'],
        P1=8*3*initial_params['blockSize']**2,
        P2=32*3*initial_params['blockSize']**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    print("✓ StereoSGBM criado")
    
    # Janelas
    print("Criando interface...")
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Comparison (Raw vs Filtered)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow('Disparity', 800, 600)
    cv2.resizeWindow('Comparison (Raw vs Filtered)', 1200, 400)
    cv2.resizeWindow('Controls', 500, 450)
    
    # Trackbars
    cv2.createTrackbar('blockSize', 'Controls', initial_params['blockSize'], 25, on_trackbar)
    cv2.createTrackbar('numDisparities', 'Controls', initial_params['numDisparities'], 20, on_trackbar)
    cv2.createTrackbar('uniquenessRatio', 'Controls', initial_params['uniquenessRatio'], 50, on_trackbar)
    cv2.createTrackbar('speckleWindow', 'Controls', initial_params['speckleWindow'], 250, on_trackbar)
    cv2.createTrackbar('speckleRange', 'Controls', initial_params['speckleRange'], 100, on_trackbar)
    cv2.createTrackbar('disp12MaxDiff', 'Controls', initial_params['disp12MaxDiff'], 100, on_trackbar)
    cv2.createTrackbar('preFilterCap', 'Controls', initial_params['preFilterCap'], 63, on_trackbar)
    cv2.createTrackbar('postFilter', 'Controls', initial_params['postFilter'], 3, on_trackbar)
    cv2.createTrackbar('medianFilter', 'Controls', initial_params['medianFilter'], 5, on_trackbar)
    
    print("✓ Controles criados")
    
    initialized = True
    
    print("\n" + "="*60)
    print("Calculando disparidade inicial...")
    compute_disparity()
    
    print("\n" + "="*60)
    print("✓✓✓ SISTEMA PRONTO ✓✓✓")
    print("="*60)
    print("\nTECLAS:")
    print("  ESC:   Sair")
    print("  SPACE: Recalcular")
    print("  S:     Salvar configuração atual (com nome automático)")
    print("  N:     Salvar com nome personalizado")
    print("  L:     Listar configurações salvas")
    print("  C:     Carregar configuração salva")
    print("="*60 + "\n")
    
    # Loop
    while True:
        key = cv2.waitKey(10) & 0xFF
        
        if key == 27:  # ESC
            break
        
        elif key == ord(' '):  # SPACE
            print("\n[MANUAL] Recalculando...")
            compute_disparity()
        
        elif key == ord('s') or key == ord('S'):  # Salvar automático
            save_configuration()
        
        elif key == ord('n') or key == ord('N'):  # Salvar com nome
            print("\n" + "="*60)
            name = input("Digite o nome da configuração: ").strip()
            if name:
                save_configuration(name)
            else:
                print("❌ Nome inválido. Use 'S' para salvar com nome automático.")
        
        elif key == ord('l') or key == ord('L'):  # Listar
            list_saved_configurations()
        
        elif key == ord('c') or key == ord('C'):  # Carregar
            configs = list_saved_configurations()
            if configs:
                try:
                    choice = int(input("Digite o número da configuração (ou 0 para cancelar): "))
                    if 1 <= choice <= len(configs):
                        config_name = configs[choice-1][0]
                        load_configuration(config_name)
                except:
                    print("❌ Entrada inválida.")
    
    cv2.destroyAllWindows()
    print("\n✓ Programa encerrado")