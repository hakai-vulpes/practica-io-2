from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

def get_aligned_face(
    rgb_image: np.ndarray,
    landmarker: vision.FaceLandmarker,
    output_size: tuple[int, int] = (512, 512),
    verbose: bool = False
) -> np.ndarray:
    h, w, _ = rgb_image.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Detect landmarks
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        if verbose:
            plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            plt.title('Image with no detected face')
            plt.show()
        raise ValueError('No se detectó ningún rostro en la imagen.')

    landmarks = result.face_landmarks[0]

    # Los landamarks están organizados por una serie de índices específicos.
    # Para los ojos, usaremos los puntos clave alrededor de ellos para calcular la alineación.
    # Derecho: 33, 133 | Izquierdo: 362, 263
    def get_pt(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

    right_eye = (get_pt(33) + get_pt(133)) / 2
    left_eye = (get_pt(362) + get_pt(263)) / 2

    # 1. Calculamos el ángulo para nivelar los ojos
    dY = left_eye[1] - right_eye[1]
    dX = left_eye[0] - right_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # 2. Calculamos la escala (distancia entre ojos vs distancia deseada)
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = output_size[0] * 0.45
    scale = desired_dist / dist

    # 3. Calculamos el punto central entre los ojos
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # 4. Obtenemos la matriz de rotación y escala
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # 5. Ajustamos la matriz para centrar la cara en el recorte
    # Ajusta '0.35' para mover la cara hacia arriba o hacia abajo
    M[0, 2] += (output_size[0] * 0.5) - eyes_center[0]
    M[1, 2] += (output_size[1] * 0.35) - eyes_center[1]

    # 6. Aplicamos la transformación
    aligned_face = cv2.warpAffine(rgb_image, M, output_size, flags=cv2.INTER_CUBIC)
    return aligned_face

def get_aligned_faces(
    *images: np.ndarray,
    landmarker_options: vision.FaceLandmarkerOptions | None = None,
    output_size: tuple[int, int] = (512, 512),
    verbose: bool = False
) -> list[np.ndarray]:
    aligned_faces = []
    # Dejamos la posibilidad de elegir las opciones del landmarker
    if landmarker_options is None:
        # Sino cargamos el modelo por defecto
        model_dir = Path('models')
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / 'face_landmarker.task'
        if not model_path.exists():
            # Descargar el modelo si no existe
            import urllib.request
            url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
            print(f'Downloading model to {model_path}...')
            urllib.request.urlretrieve(url, model_path)
            print('Download complete.')
        
        # Configuramos las opciones del landmarker
        landmarker_options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_faces=1
        )
    
    # Aprovechamos el contexto para cargar el modelo una sola vez
    with vision.FaceLandmarker.create_from_options(landmarker_options) as landmarker:
        for image_path in images:
            face = get_aligned_face(image_path, landmarker, output_size)
            aligned_faces.append(face)
    return aligned_faces
    

if __name__ == '__main__':
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='models/face_landmarker.task'),
        num_faces=1
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        image_paths = Path('images').glob('*')
        image_paths = list(image_paths)
        image1 = cv2.imread(str(image_paths[3]))
        image2 = cv2.imread(str(image_paths[4]))
        face1 = get_aligned_face(image1, landmarker)
        face2 = get_aligned_face(image2, landmarker)

    if face1 is not None and face2 is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Aligned Face 1')
        plt.imshow(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2)
        plt.title('Aligned Face 2')
        plt.imshow(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB))
        plt.show()
        print('Alignment complete.')