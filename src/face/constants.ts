/** Lado da imagem de entrada do MobileFaceNet (tensor `[1, 112, 112, 3]`). */
export const MODEL_INPUT_SIZE = 112;

/** Tamanho do vetor de saida do MobileFaceNet (tensor `[1, 192]`). */
export const EMBEDDING_SIZE = 192;

/** Normalizacao de entrada do modelo: `(pixel - IMAGE_MEAN) / IMAGE_STD`. */
export const IMAGE_MEAN = 127.5;
export const IMAGE_STD = 128.0;

/**
 * Similaridade de cosseno minima para considerar que dois rostos sao a mesma
 * pessoa.
 *
 * Calibrado sobre as fotos de IMAGENS_TESTE (23 rostos, 57 pares da mesma
 * pessoa e 196 pares de pessoas diferentes), com o alinhamento ligado:
 *
 *   menor similaridade entre MESMA pessoa ....... 0.784
 *   maior similaridade entre pessoas DIFERENTES . 0.448
 *
 * 0.62 fica quase no meio dessa folga, e qualquer valor entre ~0.50 e ~0.75
 * classifica tudo certo. Rode `npm run verify:math` e o teste offline em
 * scripts/offline-eval antes de mexer aqui.
 */
export const MATCH_THRESHOLD = 0.62;

/**
 * Folga em volta da caixa do ML Kit para o quadrado desenhado na tela.
 * Vale so para a exibicao: o recorte que vai ao modelo e posicionado pelos
 * olhos, nao por esta caixa.
 */
export const CROP_MARGIN = 1.25;

/**
 * Pontos de referencia do ArcFace para a entrada 112x112: onde os olhos
 * precisam cair depois do alinhamento. Sao os mesmos valores usados no treino
 * do MobileFaceNet.
 *
 * Alinhar por esses pontos nao e detalhe de acabamento: medido nas fotos de
 * teste, recortar so pela caixa do detector deixa as similaridades de "mesma
 * pessoa" e "pessoas diferentes" SOBREPOSTAS (nenhum limiar separa as duas).
 * Com alinhamento a separacao vira uma folga de ~0.34.
 */
export const ARCFACE_RIGHT_EYE_X = 38.2946;
export const ARCFACE_RIGHT_EYE_Y = 51.6963;
export const ARCFACE_LEFT_EYE_X = 73.5318;
export const ARCFACE_LEFT_EYE_Y = 51.5014;

/** Distancia entre os olhos no gabarito, como fracao do lado (112). */
export const ARCFACE_EYE_SPAN =
  (ARCFACE_LEFT_EYE_X - ARCFACE_RIGHT_EYE_X) / MODEL_INPUT_SIZE;

/** Ponto medio dos olhos no gabarito, como fracao do lado. */
export const ARCFACE_EYE_MID_X =
  (ARCFACE_RIGHT_EYE_X + ARCFACE_LEFT_EYE_X) / 2 / MODEL_INPUT_SIZE;
export const ARCFACE_EYE_MID_Y =
  (ARCFACE_RIGHT_EYE_Y + ARCFACE_LEFT_EYE_Y) / 2 / MODEL_INPUT_SIZE;

/**
 * Lado do buffer intermediario buscado do frame antes do alinhamento.
 *
 * O resize-plugin so recorta, escala, gira em multiplos de 90 e espelha — nao
 * faz warp afim. Entao o recorte sai numa regiao maior, eixo-alinhada, e o
 * warp final acontece em JS. Medido: 192 empata com o warp direto sobre a
 * imagem original; 112 ja perde um pouco de folga.
 */
export const REGION_SIZE = 192;

/**
 * Quanto a regiao buscada e maior que o recorte final. Precisa passar de
 * sqrt(2) para caber o quadrado girado em qualquer angulo, com folga para o
 * rosto nao estar perfeitamente centrado.
 */
export const REGION_SCALE = 1.6;

/**
 * Lado minimo (em pixels do frame) que um rosto precisa ter para valer a pena
 * tentar reconhecer. Abaixo disso o recorte fica borrado demais e o resultado
 * seria ruido -> marcamos como "nao foi possivel ler" (amarelo).
 */
export const MIN_FACE_PIXELS = 64;

/** Angulos maximos de cabeca aceitos; alem disso o rosto vira "ilegivel". */
export const MAX_YAW_ANGLE = 45;
export const MAX_PITCH_ANGLE = 40;

/** Quantas vezes por segundo rodar deteccao + reconhecimento. */
export const DETECTION_FPS = 8;

/**
 * De quantos em quantos ciclos de deteccao um rosto ja identificado e
 * reprocessado. Com `DETECTION_FPS = 8`, 16 ciclos ~= 2 segundos.
 */
export const RECHECK_EVERY_TICKS = 16;
