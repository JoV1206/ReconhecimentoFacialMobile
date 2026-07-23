"""
Deteccao em ladrilhos: roda o BlazeFace em pedacos ampliados da imagem.

O detector "short range" so enxerga bem rostos que ocupam boa parte do quadro.
Numa foto de grupo isso falha; ampliar a imagem inteira nao resolve porque o
modelo redimensiona a entrada para 128x128 de qualquer jeito. Rodar por ladrilho
faz cada rosto ocupar area suficiente, com confianca alta (sem lixo).
"""
import numpy as np
import cv2
from pipeline import detect, KP_NOSE


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect_tiled(image_bgr, tiles=(3, 2), overlap=0.35, min_confidence=0.45,
                 tile_upscale=2, iou_threshold=0.3):
    H, W = image_bgr.shape[:2]
    cols, rows = tiles
    tw, th = W / cols, H / rows
    ox, oy = tw * overlap, th * overlap

    found = []
    # Passo 1: imagem inteira (pega rostos grandes)
    found.extend(detect(image_bgr, min_confidence=min_confidence, upscale=1))

    # Passo 2: cada ladrilho ampliado
    for r in range(rows):
        for c in range(cols):
            x0 = int(max(0, c * tw - ox))
            y0 = int(max(0, r * th - oy))
            x1 = int(min(W, (c + 1) * tw + ox))
            y1 = int(min(H, (r + 1) * th + oy))
            tile = image_bgr[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            for f in detect(tile, min_confidence=min_confidence, upscale=tile_upscale):
                bx, by, bw, bh = f["box"]
                found.append({
                    "box": (bx + x0, by + y0, bw, bh),
                    "kps": f["kps"] + np.array([x0, y0], dtype=np.float64),
                    "score": f["score"],
                })

    # Passo 3: NMS por IoU, mantendo o de maior score.
    # So IoU nao basta: a deteccao da imagem inteira sai bem maior que a do
    # ladrilho para o mesmo rosto, entao a sobreposicao relativa fica baixa.
    # Suprimir tambem quando um centro cai dentro da outra caixa resolve isso.
    def _center_inside(a, b):
        cx, cy = a[0] + a[2] / 2, a[1] + a[3] / 2
        return b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]

    found.sort(key=lambda f: -f["score"])
    kept = []
    for f in found:
        duplicate = any(
            _iou(f["box"], k["box"]) >= iou_threshold
            or _center_inside(f["box"], k["box"])
            or _center_inside(k["box"], f["box"])
            for k in kept
        )
        if not duplicate:
            kept.append(f)

    kept.sort(key=lambda f: f["kps"][KP_NOSE][0])
    return kept

