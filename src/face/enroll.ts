import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { detectFaces } from 'react-native-vision-camera-face-detector';
import { decode as decodeJpeg } from 'jpeg-js';
import { toByteArray } from 'base64-js';
import type { TensorflowModel } from 'react-native-fast-tflite';

import { REGION_SIZE } from './constants';
import { isValidEmbedding, l2Normalize, warpRegionToTensor } from './embedding';
import { eyeAlignedRegion } from './geometry';

/**
 * Maior lado usado ao normalizar a foto antes da deteccao. Fotos de celular
 * chegam com 4000px ou mais, o que gasta memoria a toa: o ML Kit nao ganha
 * precisao acima disso e o recorte final tem so 112px.
 */
const MAX_WORKING_SIZE = 1200;

export type EnrollResult =
  | { ok: true; embedding: number[]; normalizedUri: string; faceCropUri: string }
  | {
      ok: false;
      reason: 'no-face' | 'multiple-faces' | 'too-small' | 'no-eyes' | 'decode-failed';
    };

/**
 * Reduz a imagem e reescreve o arquivo para que os pixels fiquem alinhados com
 * o que o detector enxerga.
 *
 * O detector de imagem estatica desta lib usa `BitmapFactory.decodeFile` com
 * rotacao 0, ou seja, ignora o EXIF. Se a foto vier de um celular na vertical
 * (EXIF "rotate 90"), as caixas sairiam giradas em relacao ao arquivo original.
 * Passar pelo ImageManipulator "assa" a rotacao nos pixels e remove o EXIF.
 */
async function normalizeImage(uri: string): Promise<{ uri: string; width: number; height: number }> {
  const original = await ImageManipulator.manipulate(uri).renderAsync();

  const longestSide = Math.max(original.width, original.height);
  const context = ImageManipulator.manipulate(original);
  if (longestSide > MAX_WORKING_SIZE) {
    const ratio = MAX_WORKING_SIZE / longestSide;
    context.resize({
      width: Math.round(original.width * ratio),
      height: Math.round(original.height * ratio),
    });
  }

  const rendered = await context.renderAsync();
  const saved = await rendered.saveAsync({ format: SaveFormat.JPEG, compress: 0.92 });
  return { uri: saved.uri, width: rendered.width, height: rendered.height };
}

/**
 * Detecta o rosto numa foto, recorta, roda o MobileFaceNet e devolve o
 * embedding normalizado pronto para ser gravado no banco.
 */
export async function computeEmbeddingFromPhoto(
  model: TensorflowModel,
  photoUri: string
): Promise<EnrollResult> {
  const normalized = await normalizeImage(photoUri);

  const faces = await detectFaces({
    image: { uri: normalized.uri },
    options: {
      performanceMode: 'accurate',
      // Mesma exigencia do caminho ao vivo: sem os olhos nao ha alinhamento, e
      // sem alinhamento o embedding gravado nao seria comparavel.
      landmarkMode: 'all',
      contourMode: 'none',
      classificationMode: 'none',
      minFaceSize: 0.1,
    },
  });

  if (faces.length === 0) return { ok: false, reason: 'no-face' };
  if (faces.length > 1) return { ok: false, reason: 'multiple-faces' };

  const landmarks = faces[0].landmarks;
  const rightEye = landmarks?.RIGHT_EYE;
  const leftEye = landmarks?.LEFT_EYE;
  if (!rightEye || !leftEye) return { ok: false, reason: 'no-eyes' };

  const region = eyeAlignedRegion(rightEye, leftEye, normalized.width, normalized.height);
  if (!region) return { ok: false, reason: 'no-eyes' };
  if (region.width < REGION_SIZE / 2) return { ok: false, reason: 'too-small' };

  const cropContext = ImageManipulator.manipulate(normalized.uri);
  cropContext.crop({
    originX: region.x,
    originY: region.y,
    width: region.width,
    height: region.height,
  });
  cropContext.resize({ width: REGION_SIZE, height: REGION_SIZE });

  const croppedRef = await cropContext.renderAsync();
  const cropped = await croppedRef.saveAsync({
    format: SaveFormat.JPEG,
    compress: 1,
    base64: true,
  });

  if (!cropped.base64) return { ok: false, reason: 'decode-failed' };

  // `formatAsRGBA: false` devolve 3 canais, exatamente o layout que o modelo espera.
  const decoded = decodeJpeg(toByteArray(cropped.base64), {
    useTArray: true,
    formatAsRGBA: false,
  });

  if (decoded.data.length < REGION_SIZE * REGION_SIZE * 3) {
    return { ok: false, reason: 'decode-failed' };
  }

  // Olhos em coordenadas do buffer recortado.
  const toRegion = REGION_SIZE / region.width;
  const tensor = warpRegionToTensor(
    decoded.data,
    REGION_SIZE,
    (rightEye.x - region.x) * toRegion,
    (rightEye.y - region.y) * toRegion,
    (leftEye.x - region.x) * toRegion,
    (leftEye.y - region.y) * toRegion
  );
  if (tensor == null) return { ok: false, reason: 'no-eyes' };

  const output = model.runSync([tensor]);
  const raw = output[0];
  if (!isValidEmbedding(raw)) return { ok: false, reason: 'decode-failed' };

  return {
    ok: true,
    embedding: l2Normalize(raw),
    normalizedUri: normalized.uri,
    faceCropUri: cropped.uri,
  };
}

export function describeEnrollFailure(reason: Exclude<EnrollResult, { ok: true }>['reason']): string {
  switch (reason) {
    case 'no-face':
      return 'Nenhum rosto foi encontrado na foto. Use uma imagem nitida, de frente e com boa iluminacao.';
    case 'multiple-faces':
      return 'A foto tem mais de um rosto. Escolha uma imagem com apenas a pessoa a ser cadastrada.';
    case 'too-small':
      return 'O rosto esta pequeno demais na foto. Use uma imagem mais proxima do rosto.';
    case 'no-eyes':
      return 'Nao foi possivel localizar os dois olhos. Use uma foto de frente, sem oculos escuros e com o rosto desobstruido.';
    default:
      return 'Nao foi possivel processar a imagem. Tente outra foto.';
  }
}
