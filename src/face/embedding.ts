import {
  ARCFACE_LEFT_EYE_X,
  ARCFACE_LEFT_EYE_Y,
  ARCFACE_RIGHT_EYE_X,
  ARCFACE_RIGHT_EYE_Y,
  EMBEDDING_SIZE,
  IMAGE_MEAN,
  IMAGE_STD,
  MODEL_INPUT_SIZE,
} from './constants';

/**
 * Normalizacao usada no treino deste MobileFaceNet: `(pixel - 127.5) / 128`,
 * levando 0..255 para aproximadamente -1..1. Os valores vem do projeto de
 * origem do modelo (IMAGE_MEAN = 127.5, IMAGE_STD = 128).
 *
 * Recebe um buffer RGB entrelacado (R,G,B,R,G,B,...) de 112x112.
 */
export function rgbBytesToTensor(rgb: Uint8Array): Float32Array {
  'worklet';
  const pixels = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3;
  const tensor = new Float32Array(pixels);
  for (let i = 0; i < pixels; i++) {
    tensor[i] = (rgb[i] - IMAGE_MEAN) / IMAGE_STD;
  }
  return tensor;
}

/**
 * Alinha o rosto e ja devolve o tensor de entrada do modelo.
 *
 * Recebe o buffer RGB quadrado (`regionSize` x `regionSize`) que o
 * resize-plugin extraiu do frame, mais a posicao dos olhos DENTRO desse buffer,
 * e aplica a transformada de similaridade que leva os olhos para os pontos do
 * gabarito ArcFace. Sai direto em 112x112 normalizado.
 *
 * Percorre o destino e amostra a origem (bilinear), que e a forma correta de
 * fazer warp: percorrer a origem deixaria buracos no destino.
 *
 * Sem trigonometria: a similaridade que leva o vetor entre olhos do gabarito de
 * volta ao vetor entre olhos medido e a razao complexa `medido / gabarito`.
 */
export function warpRegionToTensor(
  region: Uint8Array,
  regionSize: number,
  rightEyeX: number,
  rightEyeY: number,
  leftEyeX: number,
  leftEyeY: number
): Float32Array | null {
  'worklet';
  const measuredX = leftEyeX - rightEyeX;
  const measuredY = leftEyeY - rightEyeY;

  const templateX = ARCFACE_LEFT_EYE_X - ARCFACE_RIGHT_EYE_X;
  const templateY = ARCFACE_LEFT_EYE_Y - ARCFACE_RIGHT_EYE_Y;
  const templateNormSq = templateX * templateX + templateY * templateY;
  if (templateNormSq <= 0) return null;

  // razao complexa medido/gabarito -> escala + rotacao inversas de uma vez
  const ratioRe = (measuredX * templateX + measuredY * templateY) / templateNormSq;
  const ratioIm = (measuredY * templateX - measuredX * templateY) / templateNormSq;

  // Checagem de finitude sem `isFinite`: o worklets-core nao consegue
  // compartilhar funcoes globais comuns com o runtime paralelo e lanca
  // "Regular javascript function 'isFinite' cannot be shared".
  // `x * 0 === 0` so vale para numeros finitos (NaN*0 e Inf*0 dao NaN).
  if (!(ratioRe * 0 === 0) || !(ratioIm * 0 === 0)) return null;

  const tensor = new Float32Array(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3);
  const maxCoord = regionSize - 1;
  let out = 0;

  for (let v = 0; v < MODEL_INPUT_SIZE; v++) {
    const dy = v - ARCFACE_RIGHT_EYE_Y;
    for (let u = 0; u < MODEL_INPUT_SIZE; u++) {
      const dx = u - ARCFACE_RIGHT_EYE_X;

      let sx = rightEyeX + ratioRe * dx - ratioIm * dy;
      let sy = rightEyeY + ratioIm * dx + ratioRe * dy;

      // Fora da regiao: repete a borda, em vez de gerar preto que o modelo
      // interpretaria como parte do rosto.
      if (sx < 0) sx = 0;
      else if (sx > maxCoord) sx = maxCoord;
      if (sy < 0) sy = 0;
      else if (sy > maxCoord) sy = maxCoord;

      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const x1 = x0 < maxCoord ? x0 + 1 : x0;
      const y1 = y0 < maxCoord ? y0 + 1 : y0;
      const fx = sx - x0;
      const fy = sy - y0;

      const rowTop = y0 * regionSize;
      const rowBottom = y1 * regionSize;
      const iTopLeft = (rowTop + x0) * 3;
      const iTopRight = (rowTop + x1) * 3;
      const iBottomLeft = (rowBottom + x0) * 3;
      const iBottomRight = (rowBottom + x1) * 3;

      const wTopLeft = (1 - fx) * (1 - fy);
      const wTopRight = fx * (1 - fy);
      const wBottomLeft = (1 - fx) * fy;
      const wBottomRight = fx * fy;

      for (let c = 0; c < 3; c++) {
        const value =
          region[iTopLeft + c] * wTopLeft +
          region[iTopRight + c] * wTopRight +
          region[iBottomLeft + c] * wBottomLeft +
          region[iBottomRight + c] * wBottomRight;
        tensor[out++] = (value - IMAGE_MEAN) / IMAGE_STD;
      }
    }
  }

  return tensor;
}

/**
 * O MobileFaceNet nao normaliza a saida, entao fazemos isso aqui. Com vetores
 * unitarios a similaridade de cosseno vira um simples produto escalar.
 */
export function l2Normalize(vector: ArrayLike<number>): number[] {
  'worklet';
  let sum = 0;
  for (let i = 0; i < vector.length; i++) {
    sum += vector[i] * vector[i];
  }
  const norm = Math.sqrt(sum);
  if (norm === 0) {
    const zeros: number[] = [];
    for (let i = 0; i < vector.length; i++) zeros.push(0);
    return zeros;
  }
  const out: number[] = [];
  for (let i = 0; i < vector.length; i++) {
    out.push(vector[i] / norm);
  }
  return out;
}

/** Produto escalar entre dois embeddings ja normalizados (= cosseno). */
export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  'worklet';
  const length = Math.min(a.length, b.length);
  let dot = 0;
  for (let i = 0; i < length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

/**
 * Confere se o modelo devolveu um vetor com o tamanho esperado.
 *
 * O `runSync` do fast-tflite e tipado com a uniao de todos os TypedArray
 * possiveis (incluindo os de bigint). Como este modelo declara saida float32,
 * o type guard estreita para um array numerico depois da checagem.
 */
export function isValidEmbedding(
  vector: ArrayLike<number | bigint> | undefined | null
): vector is ArrayLike<number> {
  'worklet';
  return vector != null && vector.length === EMBEDDING_SIZE;
}
