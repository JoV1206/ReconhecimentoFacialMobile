import type { FaceStatus } from './types';

export const colors = {
  background: '#0B1120',
  surface: '#151E31',
  surfaceAlt: '#1E293B',
  border: '#2C3A52',
  text: '#F8FAFC',
  textMuted: '#94A3B8',
  accent: '#38BDF8',
  danger: '#F87171',

  /** Pessoa reconhecida. */
  known: '#22C55E',
  /** Rosto legivel, mas fora do banco. */
  unknown: '#EF4444',
  /** Rosto detectado que nao deu para ler. */
  unreadable: '#FACC15',
} as const;

export function statusColor(status: FaceStatus): string {
  switch (status) {
    case 'known':
      return colors.known;
    case 'unknown':
      return colors.unknown;
    default:
      return colors.unreadable;
  }
}
