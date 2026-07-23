import type { Orientation } from 'react-native-vision-camera';

import {
  ARCFACE_EYE_MID_X,
  ARCFACE_EYE_MID_Y,
  ARCFACE_EYE_SPAN,
  REGION_SCALE,
} from './constants';

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

/**
 * Quantos graus (sentido horario) o buffer bruto do frame precisa girar para
 * ficar "em pe".
 *
 * `frame.orientation` da VisionCamera e o inverso do `rotationDegrees` do
 * CameraX (veja `Frame.getOrientation()` no Android: ele faz
 * `Orientation.fromRotationDegrees(degrees).reversed()`). Como o ML Kit recebe
 * justamente o `rotationDegrees`, desfazer essa inversao aqui nos da o mesmo
 * espaco de coordenadas em que as caixas de rosto sao devolvidas.
 */
export function orientationToUprightRotation(orientation: Orientation): number {
  'worklet';
  switch (orientation) {
    case 'landscape-left':
      return 270;
    case 'portrait-upside-down':
      return 180;
    case 'landscape-right':
      return 90;
    default:
      return 0;
  }
}

/** Formato aceito pela opcao `rotation` do vision-camera-resize-plugin. */
export function rotationToPluginValue(rotation: number): '0deg' | '90deg' | '180deg' | '270deg' {
  'worklet';
  if (rotation === 90) return '90deg';
  if (rotation === 180) return '180deg';
  if (rotation === 270) return '270deg';
  return '0deg';
}

/** Dimensoes do frame depois de girado para ficar em pe. */
export function uprightFrameSize(
  frameWidth: number,
  frameHeight: number,
  rotation: number
): { width: number; height: number } {
  'worklet';
  const swapped = rotation === 90 || rotation === 270;
  return swapped
    ? { width: frameHeight, height: frameWidth }
    : { width: frameWidth, height: frameHeight };
}

/**
 * Converte um retangulo do espaco "em pe" (onde o ML Kit devolve as caixas) de
 * volta para o espaco do buffer bruto, que e o que o resize-plugin espera na
 * opcao `crop` (ele recorta antes de girar).
 */
export function uprightRectToFrameRect(
  rect: Rect,
  rotation: number,
  frameWidth: number,
  frameHeight: number
): Rect {
  'worklet';
  const { x, y, width, height } = rect;
  switch (rotation) {
    case 90:
      return { x: y, y: frameHeight - (x + width), width: height, height: width };
    case 180:
      return {
        x: frameWidth - (x + width),
        y: frameHeight - (y + height),
        width,
        height,
      };
    case 270:
      return { x: frameWidth - (y + height), y: x, width: height, height: width };
    default:
      return { x, y, width, height };
  }
}

/**
 * Transforma a caixa do ML Kit num quadrado com margem, ja limitado as bordas
 * do frame. O MobileFaceNet espera entrada quadrada; recortar retangular e
 * depois esticar para 112x112 distorceria o rosto e degradaria o embedding.
 */
export function toSquareFaceBox(
  bounds: Rect,
  margin: number,
  uprightWidth: number,
  uprightHeight: number
): Rect | null {
  'worklet';
  const centerX = bounds.x + bounds.width / 2;
  const centerY = bounds.y + bounds.height / 2;
  const side = Math.max(bounds.width, bounds.height) * margin;

  let half = side / 2;
  // Encolhe o quadrado ate caber inteiro no frame, mantendo o centro do rosto.
  half = Math.min(half, centerX, centerY, uprightWidth - centerX, uprightHeight - centerY);
  if (half <= 0) return null;

  // Origem e lado precisam ser pares: o recorte acontece no buffer YUV 4:2:0,
  // onde os canais de cor sao subamostrados de 2 em 2 pixels. Como a rotacao
  // apenas troca/espelha os eixos, valores pares aqui continuam pares no buffer.
  const size = Math.floor(half) * 2;
  if (size <= 0) return null;

  const x = Math.floor((centerX - half) / 2) * 2;
  const y = Math.floor((centerY - half) / 2) * 2;
  if (x < 0 || y < 0 || x + size > uprightWidth || y + size > uprightHeight) return null;

  return { x, y, width: size, height: size };
}

/**
 * Calcula a regiao quadrada a ser buscada do frame para depois alinhar o rosto.
 *
 * A escala vem da distancia entre os olhos, que e a medida mais estavel de um
 * rosto: nao depende de quanto o detector "aperta" a caixa nem de o queixo
 * estar cortado. O centro tambem sai dos olhos, colocando-os no mesmo ponto
 * relativo do gabarito ArcFace.
 *
 * A regiao sai maior que o recorte final (REGION_SCALE) para que o quadrado
 * girado caiba nela em qualquer angulo de inclinacao da cabeca.
 *
 * Devolve `null` quando os olhos estao colados demais (rosto de perfil ou
 * deteccao ruim) — nesse caso o app marca o rosto como ilegivel.
 */
export function eyeAlignedRegion(
  rightEye: Point,
  leftEye: Point,
  uprightWidth: number,
  uprightHeight: number
): Rect | null {
  'worklet';
  const dx = leftEye.x - rightEye.x;
  const dy = leftEye.y - rightEye.y;
  const eyeDistance = Math.sqrt(dx * dx + dy * dy);
  if (!(eyeDistance > 1)) return null;

  const targetSide = eyeDistance / ARCFACE_EYE_SPAN;
  const eyeMidX = (rightEye.x + leftEye.x) / 2;
  const eyeMidY = (rightEye.y + leftEye.y) / 2;

  // Centro do recorte final, deduzido de onde os olhos devem cair nele.
  const faceCenterX = eyeMidX + (0.5 - ARCFACE_EYE_MID_X) * targetSide;
  const faceCenterY = eyeMidY + (0.5 - ARCFACE_EYE_MID_Y) * targetSide;

  // Lado par (subamostragem do YUV 4:2:0) e que caiba no frame.
  let size = Math.floor(Math.min(targetSide * REGION_SCALE, uprightWidth, uprightHeight) / 2) * 2;
  if (size < 2) return null;

  // Desloca para dentro do frame em vez de encolher: encolher mudaria a escala
  // e desalinharia o rosto em relacao ao gabarito.
  let x = Math.round(faceCenterX - size / 2);
  let y = Math.round(faceCenterY - size / 2);
  x = Math.max(0, Math.min(x, uprightWidth - size));
  y = Math.max(0, Math.min(y, uprightHeight - size));
  x -= x % 2;
  y -= y % 2;

  return { x, y, width: size, height: size };
}

/**
 * Mapeia coordenadas do frame em pe para coordenadas da view, replicando o
 * `resizeMode="cover"` da preview (escala pelo maior fator e centraliza).
 * A camera frontal e espelhada na tela, entao o eixo X e invertido.
 */
export function uprightRectToViewRect(
  rect: Rect,
  uprightWidth: number,
  uprightHeight: number,
  viewWidth: number,
  viewHeight: number,
  mirrored: boolean
): Rect {
  'worklet';
  const scale = Math.max(viewWidth / uprightWidth, viewHeight / uprightHeight);
  const offsetX = (viewWidth - uprightWidth * scale) / 2;
  const offsetY = (viewHeight - uprightHeight * scale) / 2;

  const width = rect.width * scale;
  const height = rect.height * scale;
  const y = rect.y * scale + offsetY;
  const x = mirrored
    ? viewWidth - (rect.x * scale + offsetX) - width
    : rect.x * scale + offsetX;

  return { x, y, width, height };
}
