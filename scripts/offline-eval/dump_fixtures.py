"""
Gera fixtures para conferir a implementacao TypeScript contra a Python.

Para cada rosto de teste grava:
  - o retangulo da regiao (saida esperada de eyeAlignedRegion)
  - o buffer RGB 192x192 dessa regiao (entrada de warpRegionToTensor)
  - a posicao dos olhos dentro do buffer
  - o tensor 112x112x3 esperado (saida de warpRegionToTensor)

O lado Node carrega os .ts de verdade e compara numero a numero.
"""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tiled_detection import detect_tiled
from pipeline import (detect, MODEL_INPUT_SIZE,
                            KP_RIGHT_EYE, KP_LEFT_EYE,
                            ARCFACE_RIGHT_EYE, ARCFACE_LEFT_EYE)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\app_expo\APP_Cam_RECONHECIMENTOFACIAL\IMAGENS_TESTE"
OUT = os.path.join(HERE, "fixtures")
os.makedirs(OUT, exist_ok=True)

IMAGE_MEAN, IMAGE_STD = 127.5, 128.0
REGION_SIZE = 192
REGION_SCALE = 1.6
EYE_SPAN = (ARCFACE_LEFT_EYE[0] - ARCFACE_RIGHT_EYE[0]) / MODEL_INPUT_SIZE
EYE_MID_X = (ARCFACE_RIGHT_EYE[0] + ARCFACE_LEFT_EYE[0]) / 2 / MODEL_INPUT_SIZE
EYE_MID_Y = (ARCFACE_RIGHT_EYE[1] + ARCFACE_LEFT_EYE[1]) / 2 / MODEL_INPUT_SIZE


def eye_aligned_region(re, le, W, H):
    """Porta da mesma logica de geometry.ts::eyeAlignedRegion."""
    dx, dy = le[0] - re[0], le[1] - re[1]
    d = float(np.sqrt(dx * dx + dy * dy))
    if not (d > 1):
        return None
    target = d / EYE_SPAN
    mx, my = (re[0] + le[0]) / 2, (re[1] + le[1]) / 2
    cx = mx + (0.5 - EYE_MID_X) * target
    cy = my + (0.5 - EYE_MID_Y) * target

    size = int(np.floor(min(target * REGION_SCALE, W, H) / 2)) * 2
    if size < 2:
        return None
    x = int(round(cx - size / 2))
    y = int(round(cy - size / 2))
    x = max(0, min(x, W - size))
    y = max(0, min(y, H - size))
    x -= x % 2
    y -= y % 2
    return {"x": x, "y": y, "width": size, "height": size}


def warp_to_tensor(region_rgb, region_size, rex, rey, lex, ley):
    """Porta da mesma logica de embedding.ts::warpRegionToTensor."""
    mx, my = lex - rex, ley - rey
    tx = ARCFACE_LEFT_EYE[0] - ARCFACE_RIGHT_EYE[0]
    ty = ARCFACE_LEFT_EYE[1] - ARCFACE_RIGHT_EYE[1]
    tn = tx * tx + ty * ty
    rre = (mx * tx + my * ty) / tn
    rim = (my * tx - mx * ty) / tn

    tensor = np.zeros(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3, np.float32)
    maxc = region_size - 1
    o = 0
    for v in range(MODEL_INPUT_SIZE):
        dy = v - ARCFACE_RIGHT_EYE[1]
        for u in range(MODEL_INPUT_SIZE):
            dx = u - ARCFACE_RIGHT_EYE[0]
            sx = rex + rre * dx - rim * dy
            sy = rey + rim * dx + rre * dy
            sx = 0.0 if sx < 0 else (maxc if sx > maxc else sx)
            sy = 0.0 if sy < 0 else (maxc if sy > maxc else sy)
            x0, y0 = int(np.floor(sx)), int(np.floor(sy))
            x1 = x0 + 1 if x0 < maxc else x0
            y1 = y0 + 1 if y0 < maxc else y0
            fx, fy = sx - x0, sy - y0
            for c in range(3):
                val = (region_rgb[y0, x0, c] * (1 - fx) * (1 - fy) +
                       region_rgb[y0, x1, c] * fx * (1 - fy) +
                       region_rgb[y1, x0, c] * (1 - fx) * fy +
                       region_rgb[y1, x1, c] * fx * fy)
                tensor[o] = (val - IMAGE_MEAN) / IMAGE_STD
                o += 1
    return tensor


cases = []


def emit(name, img, kps):
    H, W = img.shape[:2]
    re, le = kps[KP_RIGHT_EYE], kps[KP_LEFT_EYE]
    region = eye_aligned_region(re, le, W, H)
    if region is None:
        print(f"  {name}: sem regiao, pulado")
        return

    patch = img[region["y"]:region["y"] + region["height"],
                region["x"]:region["x"] + region["width"]]
    resized = cv2.resize(patch, (REGION_SIZE, REGION_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    k = REGION_SIZE / region["width"]
    rex, rey = (re[0] - region["x"]) * k, (re[1] - region["y"]) * k
    lex, ley = (le[0] - region["x"]) * k, (le[1] - region["y"]) * k

    tensor = warp_to_tensor(rgb, REGION_SIZE, rex, rey, lex, ley)

    rgb.astype(np.uint8).tofile(os.path.join(OUT, f"{name}.region.bin"))
    tensor.astype(np.float32).tofile(os.path.join(OUT, f"{name}.tensor.bin"))
    cases.append({
        "name": name, "imageWidth": int(W), "imageHeight": int(H),
        "rightEye": [float(re[0]), float(re[1])],
        "leftEye": [float(le[0]), float(le[1])],
        "region": region,
        "eyesInRegion": [float(rex), float(rey), float(lex), float(ley)],
    })
    print(f"  {name}: regiao={region} olhos_buffer=({rex:.2f},{rey:.2f})-({lex:.2f},{ley:.2f})")


img = cv2.imread(os.path.join(BASE, "Foto de perfil.jpeg"))
emit("perfil", img, detect(img, 0.5, 1)[0]["kps"])

img = cv2.imread(os.path.join(BASE, "WhatsApp Image 2025-09-29 at 18.06.43.jpeg"))
emit("selfie", img, detect(img, 0.5, 1)[0]["kps"])

gimg = cv2.imread(os.path.join(BASE, "WhatsApp Image 2025-10-27 at 15.27.15.jpeg"))
for ident, f in zip(["joao", "noiva", "noivo", "coral", "menina"], detect_tiled(gimg)):
    emit(f"grupo_{ident}", gimg, f["kps"])

cap = cv2.VideoCapture(os.path.join(BASE, "6videoMbranca_CabeloPreso_S_oculos.mp4"))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ok, fr = cap.read()
cap.release()
if ok:
    fs = detect(fr, 0.5, 1)
    if len(fs) == 1:
        emit("video_meio", fr, fs[0]["kps"])

json.dump({"regionSize": REGION_SIZE, "modelInputSize": MODEL_INPUT_SIZE, "cases": cases},
          open(os.path.join(OUT, "cases.json"), "w"), indent=1)
print(f"\n{len(cases)} fixtures em {OUT}")

