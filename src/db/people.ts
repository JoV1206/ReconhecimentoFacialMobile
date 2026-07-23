import AsyncStorage from '@react-native-async-storage/async-storage';
import type { Person } from '../types';

/**
 * A versao faz parte da chave de proposito.
 *
 * Embeddings so sao comparaveis entre si se tiverem sido gerados pelo mesmo
 * pre-processamento. Quando o recorte mudou (v1 recortava pela caixa do
 * detector, v2 alinha pelos olhos), os vetores antigos viraram ruido: em vez de
 * degradar o reconhecimento silenciosamente, a chave nova comeca vazia e as
 * pessoas sao recadastradas.
 */
const STORAGE_KEY = '@reconhecimento-facial/people/v2-aligned';

function isPerson(value: unknown): value is Person {
  if (typeof value !== 'object' || value === null) return false;
  const p = value as Partial<Person>;
  return (
    typeof p.id === 'string' &&
    typeof p.name === 'string' &&
    typeof p.birthDate === 'string' &&
    Array.isArray(p.embedding) &&
    p.embedding.length > 0
  );
}

export async function loadPeople(): Promise<Person[]> {
  const raw = await AsyncStorage.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isPerson);
  } catch {
    // Se o JSON corromper, e melhor comecar vazio do que travar a camera.
    return [];
  }
}

async function savePeople(people: Person[]): Promise<void> {
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(people));
}

export async function addPerson(person: Person): Promise<Person[]> {
  const people = await loadPeople();
  const updated = [...people, person];
  await savePeople(updated);
  return updated;
}

/** Substitui a pessoa de mesmo `id`, preservando a posicao na lista. */
export async function updatePerson(person: Person): Promise<Person[]> {
  const people = await loadPeople();
  const updated = people.map((p) => (p.id === person.id ? person : p));
  await savePeople(updated);
  return updated;
}

export async function removePerson(id: string): Promise<Person[]> {
  const people = await loadPeople();
  const updated = people.filter((p) => p.id !== id);
  await savePeople(updated);
  return updated;
}

/** Idade em anos completos, ou `null` se a data for invalida. */
export function calculateAge(birthDate: string): number | null {
  const birth = new Date(`${birthDate}T00:00:00`);
  if (Number.isNaN(birth.getTime())) return null;

  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
    age--;
  }
  return age >= 0 ? age : null;
}

/** Converte `YYYY-MM-DD` para `DD/MM/YYYY` para exibicao. */
export function formatBirthDate(birthDate: string): string {
  const [year, month, day] = birthDate.split('-');
  if (!year || !month || !day) return birthDate;
  return `${day}/${month}/${year}`;
}
