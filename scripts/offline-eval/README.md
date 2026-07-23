# Avaliação offline do reconhecimento

Mede a qualidade do reconhecimento em fotos reais, no PC, sem precisar instalar
o app no celular. Foi com isso que o limiar de `0.62` foi calibrado e que o
alinhamento pelos olhos virou obrigatório.

## Por que existe

O pipeline do app roda dentro de um frame processor da VisionCamera, em worklets,
no Android. Lá dentro não dá para inspecionar embedding nenhum. Este harness
replica o **mesmo** pré-processamento (mesma normalização `(p-127.5)/128`, mesmo
alinhamento ArcFace, mesmo modelo `.tflite`) fora do aparelho, onde dá para medir.

**Diferença importante:** a detecção usa MediaPipe/BlazeFace em vez do ML Kit
(que só roda em Android). Ambos são da mesma família SSD e dão os mesmos
keypoints de olho, mas o enquadramento bruto varia. Como o recorte final é
posicionado pelos **olhos** e não pela caixa, essa diferença deixa de importar —
que é justamente uma das razões para alinhar.

## Como rodar

```bash
pip install mediapipe ai-edge-litert opencv-python numpy

# compara as variantes de recorte sobre as fotos de IMAGENS_TESTE/
python scripts/offline-eval/evaluate.py

# regera as fixtures usadas por `npm run verify:alignment`
python scripts/offline-eval/dump_fixtures.py
```

`evaluate.py` espera as imagens em `IMAGENS_TESTE/` na raiz do projeto.

## O que foi medido

23 rostos (2 fotos individuais, 5 pessoas numa foto de casamento, 16 frames de
2 vídeos) → 57 pares da mesma pessoa e 196 pares de pessoas diferentes.

| Variante de recorte | menor "mesma" | maior "diferente" | folga | erros @ 0.62 |
| --- | --- | --- | --- | --- |
| Caixa do detector + margem (versão 1 do app) | 0.397 | 0.701 | **−0.304** | 25 |
| Caixa reposicionada pelos olhos, sem girar | 0.354 | 0.684 | −0.330 | 14 |
| **Alinhado pelos olhos (versão atual)** | **0.784** | **0.448** | **+0.336** | **0** |

Folga negativa significa que as duas distribuições se **sobrepõem**: não existe
limiar que classifique tudo certo. Era o caso da primeira versão — ela produzia
tanto falsos positivos (João batendo com o noivo em 0.678) quanto falhas em
reconhecer a mesma pessoa em frames consecutivos do mesmo vídeo (0.397).

Com alinhamento a separação fica limpa e qualquer limiar entre ~0.50 e ~0.75
acerta tudo. `0.62` foi escolhido por ficar perto do meio.

## Relação com os testes do Node

`npm run verify:alignment` carrega os `.ts` reais de `src/face/` e confere que
eles produzem **exatamente** o mesmo tensor 112×112 que a implementação Python
daqui, para 8 rostos reais (as fixtures em `scripts/fixtures/`). É o que garante
que o código que roda no celular é o mesmo que foi medido aqui.
