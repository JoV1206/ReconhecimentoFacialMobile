import React, { useCallback, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Directory, File, Paths } from 'expo-file-system';
import type { TensorflowModel } from 'react-native-fast-tflite';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { formatBirthDate } from '../db/people';
import { computeEmbeddingFromPhoto, describeEnrollFailure } from '../face/enroll';
import { colors } from '../theme';
import type { Person } from '../types';

interface Props {
  model: TensorflowModel | undefined;
  onCancel: () => void;
  onSaved: (person: Person) => void;
  /** Quando presente, a tela edita esta pessoa em vez de criar uma nova. */
  person?: Person;
  /** Foto ja capturada (ex.: pela camera) para pre-preencher o slot. */
  initialPhotoUri?: string;
}

/** Aplica a mascara DD/MM/AAAA conforme o usuario digita. */
function maskDate(input: string): string {
  const digits = input.replace(/\D/g, '').slice(0, 8);
  if (digits.length <= 2) return digits;
  if (digits.length <= 4) return `${digits.slice(0, 2)}/${digits.slice(2)}`;
  return `${digits.slice(0, 2)}/${digits.slice(2, 4)}/${digits.slice(4)}`;
}

/** Valida a data digitada e converte para o ISO `YYYY-MM-DD` usado no banco. */
function toIsoDate(masked: string): string | null {
  const match = /^(\d{2})\/(\d{2})\/(\d{4})$/.exec(masked);
  if (!match) return null;

  const day = Number(match[1]);
  const month = Number(match[2]);
  const year = Number(match[3]);

  const date = new Date(year, month - 1, day);
  // Rejeita datas que "transbordam" (31/02 viraria 03/03) e datas futuras.
  if (
    date.getFullYear() !== year ||
    date.getMonth() !== month - 1 ||
    date.getDate() !== day ||
    date.getTime() > Date.now()
  ) {
    return null;
  }

  return `${match[3]}-${match[2]}-${match[1]}`;
}

/** Guarda a foto no diretorio do app para que sobreviva a limpeza do cache. */
function persistPhoto(sourceUri: string, personId: string): string {
  const directory = new Directory(Paths.document, 'people');
  directory.create({ intermediates: true, idempotent: true });

  const destination = new File(directory, `${personId}.jpg`);
  if (destination.exists) destination.delete();

  new File(sourceUri).copy(destination);
  return destination.uri;
}

export function AddPersonScreen({ model, onCancel, onSaved, person, initialPhotoUri }: Props) {
  const insets = useSafeAreaInsets();
  const isEditing = person != null;
  const [photoUri, setPhotoUri] = useState<string | null>(
    initialPhotoUri ?? person?.photoUri ?? null
  );
  const [name, setName] = useState(person?.name ?? '');
  const [birthDate, setBirthDate] = useState(
    person != null ? formatBirthDate(person.birthDate) : ''
  );
  const [saving, setSaving] = useState(false);

  const pickFromLibrary = useCallback(async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permissao negada', 'Autorize o acesso as fotos para escolher uma imagem.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
    });

    if (!result.canceled && result.assets.length > 0) {
      setPhotoUri(result.assets[0].uri);
    }
  }, []);

  const takePhoto = useCallback(async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permissao negada', 'Autorize o acesso a camera para tirar a foto.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 1 });
    if (!result.canceled && result.assets.length > 0) {
      setPhotoUri(result.assets[0].uri);
    }
  }, []);

  const handleSave = useCallback(async () => {
    const trimmedName = name.trim();

    if (!photoUri) {
      Alert.alert('Foto obrigatoria', 'Escolha ou tire uma foto do rosto da pessoa.');
      return;
    }
    if (trimmedName.length < 2) {
      Alert.alert('Nome invalido', 'Informe o nome da pessoa.');
      return;
    }

    const isoDate = toIsoDate(birthDate);
    if (!isoDate) {
      Alert.alert('Data invalida', 'Informe a data de nascimento no formato DD/MM/AAAA.');
      return;
    }

    // Editar so o nome/data, mantendo a mesma foto: nao precisa do modelo nem
    // de reprocessar o rosto, so atualiza os campos e reaproveita o embedding.
    const photoUnchanged = isEditing && photoUri === person!.photoUri;
    if (photoUnchanged) {
      onSaved({ ...person!, name: trimmedName, birthDate: isoDate });
      return;
    }

    if (!model) {
      Alert.alert('Modelo carregando', 'Aguarde o modelo de reconhecimento terminar de carregar.');
      return;
    }

    setSaving(true);
    try {
      const result = await computeEmbeddingFromPhoto(model, photoUri);
      if (!result.ok) {
        Alert.alert('Nao deu para cadastrar', describeEnrollFailure(result.reason));
        return;
      }

      // Ao editar, mantem id e data de criacao; ao criar, gera novos.
      const id = person?.id ?? `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      onSaved({
        id,
        name: trimmedName,
        birthDate: isoDate,
        photoUri: persistPhoto(result.normalizedUri, id),
        embedding: result.embedding,
        createdAt: person?.createdAt ?? Date.now(),
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      Alert.alert('Erro ao cadastrar', message);
    } finally {
      setSaving(false);
    }
  }, [birthDate, isEditing, model, name, onSaved, person, photoUri]);

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView
        contentContainerStyle={[
          styles.content,
          { paddingTop: insets.top + 16, paddingBottom: insets.bottom + 32 },
        ]}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.header}>
          <Text style={styles.title}>{isEditing ? 'Editar pessoa' : 'Nova pessoa'}</Text>
          <Pressable onPress={onCancel} hitSlop={12} disabled={saving}>
            <Text style={styles.cancel}>Cancelar</Text>
          </Pressable>
        </View>

        <Pressable style={styles.photoSlot} onPress={pickFromLibrary} disabled={saving}>
          {photoUri ? (
            <Image source={{ uri: photoUri }} style={styles.photo} resizeMode="cover" />
          ) : (
            <View style={styles.photoPlaceholder}>
              <Text style={styles.photoPlaceholderIcon}>+</Text>
              <Text style={styles.photoPlaceholderText}>Toque para escolher uma foto</Text>
            </View>
          )}
        </Pressable>

        <View style={styles.photoActions}>
          <Pressable style={styles.secondaryButton} onPress={pickFromLibrary} disabled={saving}>
            <Text style={styles.secondaryButtonText}>Galeria</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={takePhoto} disabled={saving}>
            <Text style={styles.secondaryButtonText}>Tirar foto</Text>
          </Pressable>
        </View>

        <Text style={styles.label}>Nome</Text>
        <TextInput
          style={styles.input}
          value={name}
          onChangeText={setName}
          placeholder="Nome completo"
          placeholderTextColor={colors.textMuted}
          autoCapitalize="words"
          editable={!saving}
        />

        <Text style={styles.label}>Data de nascimento</Text>
        <TextInput
          style={styles.input}
          value={birthDate}
          onChangeText={(text) => setBirthDate(maskDate(text))}
          placeholder="DD/MM/AAAA"
          placeholderTextColor={colors.textMuted}
          keyboardType="number-pad"
          maxLength={10}
          editable={!saving}
        />

        <Text style={styles.hint}>
          Use uma foto nitida, de frente e com apenas o rosto da pessoa. O app extrai o vetor facial
          da imagem e guarda tudo no proprio aparelho.
        </Text>

        <Pressable
          style={[styles.primaryButton, saving && styles.buttonDisabled]}
          onPress={() => void handleSave()}
          disabled={saving}
        >
          {saving ? (
            <ActivityIndicator color={colors.background} />
          ) : (
            <Text style={styles.primaryButtonText}>
              {isEditing ? 'Salvar alteracoes' : 'Salvar no banco'}
            </Text>
          )}
        </Pressable>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { paddingHorizontal: 20, gap: 14 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  title: { color: colors.text, fontSize: 24, fontWeight: '700' },
  cancel: { color: colors.accent, fontSize: 16, fontWeight: '600' },
  photoSlot: {
    height: 260,
    borderRadius: 16,
    overflow: 'hidden',
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  photo: { width: '100%', height: '100%' },
  photoPlaceholder: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 8 },
  photoPlaceholderIcon: { color: colors.accent, fontSize: 44, fontWeight: '300' },
  photoPlaceholderText: { color: colors.textMuted, fontSize: 14 },
  photoActions: { flexDirection: 'row', gap: 12 },
  secondaryButton: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
    backgroundColor: colors.surfaceAlt,
    borderWidth: 1,
    borderColor: colors.border,
  },
  secondaryButtonText: { color: colors.text, fontWeight: '600' },
  label: { color: colors.textMuted, fontSize: 13, marginTop: 6 },
  input: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 12,
    color: colors.text,
    fontSize: 16,
  },
  hint: { color: colors.textMuted, fontSize: 12, lineHeight: 18, marginTop: 4 },
  primaryButton: {
    marginTop: 10,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    backgroundColor: colors.accent,
  },
  primaryButtonText: { color: colors.background, fontSize: 16, fontWeight: '700' },
  buttonDisabled: { opacity: 0.6 },
});
