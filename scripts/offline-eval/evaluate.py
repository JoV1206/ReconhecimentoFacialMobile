"""
Avaliacao final, com 3 variantes de recorte e os dois videos separados.

  A1 "caixa crua"   : caixa do detector + margem 1.25  <- o que o app faz hoje
  A2 "caixa keypts" : caixa estimada por keypoints + margem 1.25
  B  "alinhado"     : similaridade pelos olhos (padrao ArcFace)

A1 e a comparacao mais justa com o app real, porque usa a caixa que o detector
devolve, sem heuristica minha no meio.
"""
import os, sys, cv2, numpy as np, itertools, json
sys.path.insert(0, os.path.dirname(__file__))
from tiled_detection import detect_tiled
from pipeline import (FaceModel, detect, crop_aligned, crop_app_like,
                            square_face_box, MODEL_INPUT_SIZE)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\app_expo\APP_Cam_RECONHECIMENTOFACIAL\IMAGENS_TESTE"
model = FaceModel()


def crop_raw_box(image_bgr, kps, box):
    """A1: caixa crua do detector + quadrado/margem do app."""
    H, W = image_bgr.shape[:2]
    sq = square_face_box(box[0], box[1], box[2], box[3], W, H)
    if sq is None:
        return None
    bx, by, bs, _ = sq
    crop = image_bgr[by:by + bs, bx:bx + bs]
    if crop.size == 0:
        return None
    return cv2.cvtColor(cv2.resize(crop, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                   interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


samples = []  # (identidade, rotulo, img, kps, box)


def add_single(identity, label, path):
    img = cv2.imread(path)
    faces = detect(img, min_confidence=0.5, upscale=1)
    if len(faces) != 1:
        print(f"  ! {label}: {len(faces)} rostos, pulado")
        return
    samples.append((identity, label, img, faces[0]["kps"], faces[0]["box"]))


add_single("joao", "perfil", os.path.join(BASE, "Foto de perfil.jpeg"))
add_single("selfieW", "selfie", os.path.join(BASE, "WhatsApp Image 2025-09-29 at 18.06.43.jpeg"))

group_img = cv2.imread(os.path.join(BASE, "WhatsApp Image 2025-10-27 at 15.27.15.jpeg"))
group_faces = detect_tiled(group_img)
GROUP_IDS = ["joao", "noiva", "noivo", "coral", "menina"]
assert len(group_faces) == len(GROUP_IDS)
for ident, f in zip(GROUP_IDS, group_faces):
    samples.append((ident, f"grupo-{ident}", group_img, f["kps"], f["box"]))

# Videos: identidade separada por video (nao sabemos se sao a mesma pessoa;
# as similaridades e que vao dizer).
VIDEOS = [("vid4", "4videoMbrancaS_oculos.mp4"), ("vid6", "6videoMbranca_CabeloPreso_S_oculos.mp4")]
for tag, fname in VIDEOS:
    cap = cv2.VideoCapture(os.path.join(BASE, fname))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    got = 0
    for k in range(8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 0.5) / 8))
        ok, frame = cap.read()
        if not ok:
            continue
        faces = detect(frame, min_confidence=0.5, upscale=1)
        if len(faces) == 1:
            samples.append((tag, f"{tag}_{k}", frame, faces[0]["kps"], faces[0]["box"]))
            got += 1
    cap.release()
    print(f"  {tag}: {got} frames")

print(f"\n{len(samples)} amostras")
for ident in sorted(set(s[0] for s in samples)):
    print(f"  {ident}: {sum(1 for s in samples if s[0] == ident)}")

VARIANTS = [
    ("A1 caixa crua (= app hoje)", lambda img, kps, box: crop_raw_box(img, kps, box)),
    ("A2 caixa por keypoints", lambda img, kps, box: crop_app_like(img, kps)),
    ("B  alinhado pelos olhos", lambda img, kps, box: crop_aligned(img, kps)),
]

summary = {}
for vname, crop_fn in VARIANTS:
    embeds = []
    for ident, label, img, kps, box in samples:
        c = crop_fn(img, kps, box)
        if c is not None:
            embeds.append((ident, label, model.embed(c)))

    same, diff = [], []
    for (i1, l1, e1), (i2, l2, e2) in itertools.combinations(embeds, 2):
        s = float(np.dot(e1, e2))
        (same if i1 == i2 else diff).append((s, l1, l2))
    same.sort(); diff.sort(reverse=True)
    ss, dd = np.array([x[0] for x in same]), np.array([x[0] for x in diff])

    # Melhor limiar = o que minimiza erros totais
    best_t, best_err = None, 1e9
    for t in np.arange(0.20, 0.95, 0.005):
        err = int((ss < t).sum()) + int((dd >= t).sum())
        if err < best_err:
            best_err, best_t = err, float(t)

    summary[vname] = dict(gap=float(ss.min() - dd.max()), best_t=best_t, best_err=best_err,
                          n_same=len(ss), n_diff=len(dd))

    print("\n" + "=" * 76)
    print(vname)
    print("=" * 76)
    print(f"  MESMA pessoa      : min={ss.min():.3f} media={ss.mean():.3f} max={ss.max():.3f}")
    print(f"  Pessoas DIFERENTES: min={dd.min():.3f} media={dd.mean():.3f} max={dd.max():.3f}")
    print(f"  GAP = {ss.min() - dd.max():+.3f}   (positivo = separa perfeitamente)")
    err62 = int((ss < 0.62).sum()) + int((dd >= 0.62).sum())
    print(f"  No limiar 0.62 do app: {int((ss<0.62).sum())}/{len(ss)} perdidos + "
          f"{int((dd>=0.62).sum())}/{len(dd)} falsos positivos = {err62} erros")
    print(f"  Melhor limiar possivel: {best_t:.3f} -> {best_err} erros")
    print(f"  piores DIFERENTES: " + "; ".join(f"{s:.3f} {a}~{b}" for s, a, b in diff[:3]))
    print(f"  piores MESMA     : " + "; ".join(f"{s:.3f} {a}~{b}" for s, a, b in same[:3]))

    # Os dois videos sao a mesma pessoa?
    cross = [s for s, a, b in diff if (a.startswith("vid4") and b.startswith("vid6"))
             or (a.startswith("vid6") and b.startswith("vid4"))]
    if cross:
        print(f"  vid4 x vid6 (pessoas diferentes?): media={np.mean(cross):.3f} max={max(cross):.3f}")

json.dump(summary, open(os.path.join(HERE, "final_summary.json"), "w"), indent=1)
print("\nsalvo final_summary.json")

