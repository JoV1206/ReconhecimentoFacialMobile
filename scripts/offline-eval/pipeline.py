"""
Pipeline de teste v2: usa os 6 keypoints do BlazeFace para enquadrar.

Duas variantes de recorte:
  A) "app-like": caixa quadrada estilo ML Kit + margem 1.25, igual ao app.
  B) "aligned" : alinhamento por similaridade nos olhos (padrao ArcFace), que e
                 como o MobileFaceNet foi treinado.

Comparar as duas diz se vale a pena adicionar alinhamento no app.
"""
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ai_edge_litert.interpreter import Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))
DETECTOR_PATH = os.path.join(HERE, "blaze_face_short_range.tflite")
MODEL_PATH = r"C:\app_expo\APP_Cam_RECONHECIMENTOFACIAL\assets\models\mobilefacenet.tflite"

MODEL_INPUT_SIZE = 112
IMAGE_MEAN = 127.5
IMAGE_STD = 128.0
MATCH_THRESHOLD = 0.62
CROP_MARGIN = 1.25

# Indices dos keypoints do BlazeFace
KP_RIGHT_EYE, KP_LEFT_EYE, KP_NOSE, KP_MOUTH, KP_RIGHT_EAR, KP_LEFT_EAR = range(6)

# Pontos de referencia do ArcFace para saida 112x112 (olhos apenas)
ARCFACE_RIGHT_EYE = np.array([38.2946, 51.6963])
ARCFACE_LEFT_EYE = np.array([73.5318, 51.5014])


class FaceModel:
    def __init__(self, path=MODEL_PATH):
        self.interpreter = Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

    def embed(self, face_rgb_112):
        tensor = (face_rgb_112.astype(np.float32) - IMAGE_MEAN) / IMAGE_STD
        self.interpreter.set_tensor(self.inp["index"], tensor[None, ...])
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.out["index"])[0].astype(np.float64)
        n = np.linalg.norm(raw)
        return raw / n if n > 0 else raw


_detectors = {}


def detect(image_bgr, min_confidence=0.5, upscale=1):
    """Devolve lista de dicts com box e keypoints em pixels da imagem original."""
    if min_confidence not in _detectors:
        opts = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=DETECTOR_PATH),
            min_detection_confidence=min_confidence,
        )
        _detectors[min_confidence] = mp_vision.FaceDetector.create_from_options(opts)
    detector = _detectors[min_confidence]

    h, w = image_bgr.shape[:2]
    src = image_bgr if upscale == 1 else cv2.resize(
        image_bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    out = []
    for det in res.detections:
        bb = det.bounding_box
        # keypoints vem normalizados 0..1 -> multiplica pelas dimensoes originais
        kps = np.array([[k.x * w, k.y * h] for k in det.keypoints], dtype=np.float64)
        out.append({
            "box": (bb.origin_x / upscale, bb.origin_y / upscale,
                    bb.width / upscale, bb.height / upscale),
            "kps": kps,
            "score": det.categories[0].score if det.categories else 0.0,
        })
    out.sort(key=lambda f: f["kps"][KP_NOSE][0])
    return out


def mlkit_like_box(kps):
    """
    Aproxima a caixa que o ML Kit devolveria, a partir dos keypoints.

    Usa a distancia olhos->boca como escala: num rosto frontal ela vale cerca de
    1/3 da altura do rosto. Assim o enquadramento fica estavel mesmo quando a
    caixa bruta do detector esta frouxa.
    """
    eye_center = (kps[KP_RIGHT_EYE] + kps[KP_LEFT_EYE]) / 2.0
    mouth = kps[KP_MOUTH]
    d = np.linalg.norm(mouth - eye_center)
    if d <= 0:
        return None
    size = 3.0 * d
    center = eye_center + 0.62 * (mouth - eye_center)
    return center[0] - size / 2, center[1] - size / 2, size, size


def square_face_box(x, y, w, h, W, H, margin=CROP_MARGIN):
    """Porta fiel de toSquareFaceBox() do geometry.ts."""
    cx, cy = x + w / 2.0, y + h / 2.0
    half = max(w, h) * margin / 2.0
    half = min(half, cx, cy, W - cx, H - cy)
    if half <= 0:
        return None
    size = int(np.floor(half)) * 2
    if size <= 0:
        return None
    bx = int(np.floor((cx - half) / 2)) * 2
    by = int(np.floor((cy - half) / 2)) * 2
    if bx < 0 or by < 0 or bx + size > W or by + size > H:
        return None
    return bx, by, size, size


def crop_app_like(image_bgr, kps):
    """Variante A: exatamente o que o app faz, com caixa estilo ML Kit."""
    H, W = image_bgr.shape[:2]
    base = mlkit_like_box(kps)
    if base is None:
        return None
    box = square_face_box(*base, W, H)
    if box is None:
        return None
    bx, by, bs, _ = box
    crop = image_bgr[by:by + bs, bx:bx + bs]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def crop_aligned(image_bgr, kps):
    """Variante B: alinhamento por similaridade usando os dois olhos."""
    src = np.stack([kps[KP_RIGHT_EYE], kps[KP_LEFT_EYE]])
    dst = np.stack([ARCFACE_RIGHT_EYE, ARCFACE_LEFT_EYE])

    # Transformada de similaridade a partir de 2 pontos (escala + rotacao + translacao)
    s_vec = src[1] - src[0]
    d_vec = dst[1] - dst[0]
    s_norm = np.linalg.norm(s_vec)
    if s_norm <= 0:
        return None
    scale = np.linalg.norm(d_vec) / s_norm
    angle = np.arctan2(d_vec[1], d_vec[0]) - np.arctan2(s_vec[1], s_vec[0])
    cos_a, sin_a = np.cos(angle) * scale, np.sin(angle) * scale
    M = np.array([
        [cos_a, -sin_a, dst[0][0] - (cos_a * src[0][0] - sin_a * src[0][1])],
        [sin_a, cos_a, dst[0][1] - (sin_a * src[0][0] + cos_a * src[0][1])],
    ])
    warped = cv2.warpAffine(image_bgr, M, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                            flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


def cosine(a, b):
    return float(np.dot(a, b))

