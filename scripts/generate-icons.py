"""
Gera os icones do app a partir de codigo, para nao depender de binarios opacos.

Conceito: as quatro marcas de canto sao o proprio quadrado de rastreamento que o
app desenha sobre os rostos, e dentro delas a silhueta de um rosto. Quem ja usou
o app reconhece o simbolo; quem nao usou entende que e camera + pessoa.

Rode com `python scripts/generate-icons.py`.
"""
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")

# Mesma paleta de src/theme.ts
BG = (11, 17, 32, 255)         # #0B1120
ACCENT = (56, 189, 248, 255)   # #38BDF8
FACE = (226, 240, 252, 255)    # branco levemente azulado
KNOWN = (34, 197, 94, 255)     # #22C55E

S = 1024  # resolucao de trabalho


def rounded_line(draw, xy, width, fill):
    """Linha com pontas arredondadas (PIL nao tem linecap)."""
    draw.line(xy, fill=fill, width=width)
    r = width // 2
    for (x, y) in (xy[0], xy[1]):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def draw_brackets(draw, box, arm, thickness, fill):
    """As quatro marcas de canto do quadrado de rastreamento."""
    x0, y0, x1, y1 = box
    for (cx, cy, dx, dy) in (
        (x0, y0, 1, 1),    # superior esquerdo
        (x1, y0, -1, 1),   # superior direito
        (x0, y1, 1, -1),   # inferior esquerdo
        (x1, y1, -1, -1),  # inferior direito
    ):
        rounded_line(draw, [(cx, cy), (cx + dx * arm, cy)], thickness, fill)
        rounded_line(draw, [(cx, cy), (cx, cy + dy * arm)], thickness, fill)


def draw_face(draw, cx, cy, head_r, fill):
    """Silhueta simples: cabeca + ombros. Legivel em 48px."""
    draw.ellipse([cx - head_r, cy - head_r, cx + head_r, cy + head_r], fill=fill)

    # ombros: arco grosso comecando abaixo da cabeca. As proporcoes sao
    # apertadas de proposito para a silhueta caber dentro dos colchetes — se
    # encostar neles, some a leitura de "rosto dentro do quadro".
    sw = head_r * 1.98          # semi-largura dos ombros
    sh = head_r * 1.40          # altura do arco
    top = cy + head_r * 1.30    # folga entre queixo e ombros
    draw.pieslice([cx - sw, top, cx + sw, top + sh * 2], start=180, end=360, fill=fill)


def compose(size, *, background, scale):
    """Monta o icone. `scale` encolhe o desenho para caber na zona segura."""
    img = Image.new("RGBA", (S, S), background)
    draw = ImageDraw.Draw(img)

    c = S / 2
    half = (S * 0.40) * scale
    box = (c - half, c - half, c + half, c + half)

    draw_brackets(draw, box, arm=half * 0.52, thickness=int(S * 0.055 * scale), fill=ACCENT)
    draw_face(draw, c, c - S * 0.055 * scale, head_r=S * 0.112 * scale, fill=FACE)

    # ponto verde: o estado "reconhecido", que e o objetivo do app
    dot = S * 0.038 * scale
    dx, dy = box[2], box[3]
    draw.ellipse([dx - dot, dy - dot, dx + dot, dy + dot], fill=KNOWN)

    return img.resize((size, size), Image.LANCZOS)


def save(img, name):
    path = os.path.abspath(os.path.join(ASSETS, name))
    img.save(path)
    print(f"  {name:<22} {img.size[0]}x{img.size[1]}")


print("gerando icones em assets/")

# Icone principal: fundo solido, desenho cheio
save(compose(1024, background=BG, scale=1.0), "icon.png")

# Adaptive icon do Android: fundo transparente e desenho dentro da zona segura
# (o sistema recorta ~1/3 das bordas conforme a mascara do launcher)
save(compose(1024, background=(0, 0, 0, 0), scale=0.62), "adaptive-icon.png")

# Splash: fundo transparente, backgroundColor vem do app.json
save(compose(1024, background=(0, 0, 0, 0), scale=0.80), "splash-icon.png")

# Favicon da versao web
save(compose(96, background=BG, scale=1.0), "favicon.png")

print("pronto")
