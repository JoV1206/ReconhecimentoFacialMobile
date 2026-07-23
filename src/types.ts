/** Uma pessoa cadastrada no banco local. */
export interface Person {
  id: string;
  name: string;
  /** Data de nascimento no formato ISO `YYYY-MM-DD`. */
  birthDate: string;
  /** Caminho da foto copiada para o diretorio do app. */
  photoUri: string;
  /** Embedding facial ja normalizado (L2), 192 dimensoes. */
  embedding: number[];
  createdAt: number;
}

/**
 * Resultado do reconhecimento de um rosto:
 * - `known`: bate com alguem do banco -> quadrado verde
 * - `unknown`: rosto legivel mas sem correspondencia -> quadrado vermelho
 * - `unreadable`: rosto detectado mas nao foi possivel ler -> quadrado amarelo
 */
export type FaceStatus = 'known' | 'unknown' | 'unreadable';

/** Um rosto ja convertido para coordenadas de tela, pronto para desenhar. */
export interface OverlayFace {
  /** `trackingId` do ML Kit (ou um indice, quando o tracking nao devolve id). */
  key: number;
  x: number;
  y: number;
  width: number;
  height: number;
  status: FaceStatus;
  /** Nome da pessoa, "Desconhecido" ou "Lendo...". */
  label: string;
  /** Similaridade de cosseno com a melhor correspondencia (0..1). */
  score: number;
}
