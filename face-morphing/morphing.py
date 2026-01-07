import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2

import ot


def load_face_cascade() -> cv2.CascadeClassifier:
    # Cargar el modelo preentrenado de detección de rostros
    model = 'haarcascade_frontalface_default.xml'
    cascade_path = os.path.join(cv2.data.haarcascades, model)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    # Algunas rutas pueden causar problemas
    if not face_cascade.empty():
        return face_cascade
    
    # Primero comprobamos si el archivo ya existe en local
    if os.path.exists(model):
        face_cascade = cv2.CascadeClassifier(model)
        if not face_cascade.empty():
            return face_cascade
    
    
    # Si no existe, lo copiamos desde la ruta de OpenCV a una ruta local
    import shutil
    print('Haar cascade classifier xml file not found in OpenCV data. Copying to local directory.')
    local_cascade_path = 'haarcascade_frontalface_default.xml'
    shutil.copyfile(cascade_path, local_cascade_path)

    face_cascade = cv2.CascadeClassifier(local_cascade_path)
    if face_cascade.empty():
        raise IOError('Could not load Haar cascade classifier xml file.')
        
    return face_cascade


def get_face_cascade(img: np.ndarray, size: tuple[int, int], cascade_classifier: cv2.CascadeClassifier) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = cascade_classifier.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return cv2.resize(img, size)
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    return cv2.resize(face, size)

def pad_img(img: np.ndarray, pad_size: int):
    return cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)


def build_pixel_ot(
    channel_a: np.ndarray,
    channel_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Construye lo necesario para resolver el problema de OT entre dos canales de imagen
    H, W = channel_a.shape
    # Normalizamos las intensidades
    a = (channel_a + 1e-5).ravel()
    b = (channel_b + 1e-5).ravel()
    a /= a.sum()
    b /= b.sum()

    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
    return coords, coords, a, b


def solve_ot(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reg: float = 0.01,
    max_iter: int = 5000
) -> np.ndarray:
    # Resuelve el problema de transporte óptimo entre dos distribuciones usando el algoritmo de Sinkhorn
    M = ot.dist(x, y, metric='sqeuclidean').astype(np.float64)
    M /= M.max()
    G = ot.sinkhorn(a, b, M, reg=reg, numItermax=max_iter)
    return G


def displacement_interpolation_sequence_color(x, y, G, ts, shape, color_a, color_b):
    H, W = shape
    c_a = color_a.reshape(-1, 3)
    c_b = color_b.reshape(-1, 3)
    
    # Calculamos la masa asignada a cada píxel
    mass_a = np.sum(G, axis=1)  # Por filas (origen)
    mass_b = np.sum(G, axis=0)  # Por columnas (destino)
    
    # Tomamos solo los puntos con masa significativa
    i_idx, j_idx = np.where(G > 1e-10)
    weights = G[i_idx, j_idx]
    
    frames = []
    for t in ts:
        # Interpolamos cada posición
        pos = (1 - t) * x[i_idx] + t * y[j_idx]
        
        # Obtenemos los colores correspondientes a interpolar
        # Para que no haya pérdida de color, normalizamos por la masa asignada
        color_val_a = c_a[i_idx] / (mass_a[i_idx, None] + 1e-10)
        color_val_b = c_b[j_idx] / (mass_b[j_idx, None] + 1e-10)
        colors = (1 - t) * color_val_a + t * color_val_b

        # Como con ts intermedios puede salir un punto entre píxeles cada punto se reparte
        # en los 4 píxeles más cercanos interpolando bilinealmente 
        pos_x_low = np.floor(pos[:, 0]).astype(np.int64)
        pos_y_low = np.floor(pos[:, 1]).astype(np.int64)
        pos_x_high = pos_x_low + 1
        pos_y_high = pos_y_low + 1
        fx = (pos[:, 0] - pos_x_low)[:, np.newaxis]
        fy = (pos[:, 1] - pos_y_low)[:, np.newaxis]

        acc_img = np.zeros((H, W, 3), dtype=np.float64)

        def paint(pos_x, pos_y, weight):
            mask = (pos_x >= 0) & (pos_x < W) & (pos_y >= 0) & (pos_y < H)
            np.add.at(acc_img, (pos_y[mask], pos_x[mask]), (weights[mask, None] * colors[mask] * weight[mask]))

        # Interpolación bilineal
        paint(pos_x_low , pos_y_low , (1 - fx) * (1 - fy))
        paint(pos_x_high, pos_y_low , fx       * (1 - fy))
        paint(pos_x_low , pos_y_high, (1 - fx) * fy      )
        paint(pos_x_high, pos_y_high, fx       * fy      )

        frames.append(acc_img)

    return frames


def remove_padding(img: np.ndarray, pad_size: int):
    return img[pad_size:-pad_size, pad_size:-pad_size]

def normalize_intensity(img: np.ndarray):
    img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 255
    return img.astype(np.uint8)

def sharpen_image(img: np.ndarray, alpha: float = 1.5, beta: float = -0.5):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img, alpha, blurred, beta, 0)
    return sharpened

def clean(img: np.ndarray, pad_size: int, sharpness: float = 1.0):
    img = remove_padding(img, pad_size)
    img = normalize_intensity(img)
    if sharpness > 1.0:
        img = sharpen_image(img, alpha=sharpness, beta=1-sharpness)
    return img


def process_and_morph_color(
    path_a: str,
    path_b: str,
    reg: float = 1e-3,
    max_iter: int = 1000,
    fracs: list[float] | None = None,
    sharpness: float = 1.0,
    size: tuple[int, int] = (64, 64),
    verbose: bool = False,
) -> list[np.ndarray]:
    # Función principal para cargar, procesar y realizar morphing entre los rostros en dos imágenes
    if fracs is None:
        fracs = [0.5]

    # 1. CARGA DE IMÁGENES
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    if img_a is None or img_b is None:
        if img_a is None:
            raise ValueError(f'Image at path_a "{path_a}" could not be loaded.')
        raise ValueError(f'Image at path_b "{path_b}" could not be loaded.')
    
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
    
    # 2. RECORTE DE ROSTROS
    face_cascade = load_face_cascade()
    face_a = get_face_cascade(img_a, size, face_cascade).astype(np.float64)
    face_b = get_face_cascade(img_b, size, face_cascade).astype(np.float64)
    
    # 3. PROCESAMIENTO
    # Vamos a resolver la tarea de trasporte óptimo en escala de grises, tomando como 
    # que esto representa la "masa" a transportar. Si tomaramos cada canal por separado,
    # el resultado se generarían artefactos como de "arcoíris" y no se vería bien.
    gray_a = cv2.cvtColor(face_a.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    gray_b = cv2.cvtColor(face_b.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)

    # Añadimos bordes a las imágenes, esto es porque en pasos intermedios del transporte
    # siempre tenderá a haber menos masa en los bordes al estar interpolando entre ambas
    # imágenes. Esto soluciona ese problema.
    pad_size = size[0] // 8
    p_gray_a = pad_img(gray_a, pad_size)
    p_gray_b = pad_img(gray_b, pad_size)
    p_color_a = pad_img(face_a, pad_size)
    p_color_b = pad_img(face_b, pad_size)
    
    # 4. CONSTRUCCIÓN Y RESOLUCIÓN OT
    # Como hemos dicho, se realiza sobre la imágen en escala de grises
    x, y, a, b = build_pixel_ot(p_gray_a, p_gray_b)
    G = solve_ot(x, y, a, b, reg=reg, max_iter=max_iter)

    # 5. MORPHING
    # Utilizando el plan de transporte G, generamos las imágenes intermedias a color
    raw_frames = displacement_interpolation_sequence_color(
        x, y, G, fracs, p_gray_a.shape, p_color_a, p_color_b
    )
    
    # 6. POST-PROCESADO (quitar bordes y normalizar intensidades)
    final_frames = [
        clean(
            frame,
            pad_size,
            sharpness=(1-abs(frac-0.5)*2)*(sharpness - 1) + 1
        ) for frame, frac in zip(raw_frames, fracs)
    ]
    face_a_clean = clean(p_color_a, pad_size)
    face_b_clean = clean(p_color_b, pad_size)
    
    # 6. PLOTTING
    if verbose:
        n_plots = len(fracs) + 2
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        fig.suptitle(f'Optimal Transport Morphing (reg={reg})', fontsize=16)
        
        axes[0].imshow(face_a_clean)
        axes[0].set_title('Source (t=0)')
        axes[0].axis('off')
        for i, (ax, t, morph) in enumerate(zip(axes[1:], fracs, final_frames)):
            ax.imshow(morph)
            ax.set_title(f't={t}')
            ax.axis('off')
            
        axes[-1].imshow(face_b_clean)
        axes[-1].set_title('Target (t=1)')
        axes[-1].axis('off')
        
        plt.show()
    
    return [face_a_clean, *final_frames, face_b_clean]


def visualize_sequence(frames: list[np.ndarray], **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    fig, ax = plt.subplots(**kwargs)
    im = ax.imshow(frames[0], cmap='gray')
    ax.axis('off')

    def update(frame):
        im.set_data(frames[frame])
        return [im]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100)
    plt.close()
    return HTML(ani.to_jshtml())


if __name__ == '__main__':
    image_path = Path('images')
    if not image_path.exists():
        image_path = Path('face-morphing') / 'images'
        if not image_path.exists():
            raise ValueError('Image path not found.')

    image_a_path, image_b_path, *_ = sorted(image_path.glob('*'))
    frames = process_and_morph_color(
        str(image_a_path),
        str(image_b_path),
        reg=1e-3,
        max_iter=20000,
        fracs=[0.2, 0.4, 0.6, 0.8],
        size=(64, 64),
        verbose=True
    )
    
    visualize_sequence(frames)