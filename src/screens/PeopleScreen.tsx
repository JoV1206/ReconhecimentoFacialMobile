import React from 'react';
import { Alert, FlatList, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { calculateAge, formatBirthDate } from '../db/people';
import { colors } from '../theme';
import type { Person } from '../types';

interface Props {
  people: Person[];
  onClose: () => void;
  onRemove: (id: string) => void;
  onEdit: (person: Person) => void;
}

export function PeopleScreen({ people, onClose, onRemove, onEdit }: Props) {
  const insets = useSafeAreaInsets();

  const confirmRemove = (person: Person) => {
    Alert.alert('Remover cadastro', `Remover ${person.name} do banco de dados?`, [
      { text: 'Cancelar', style: 'cancel' },
      { text: 'Remover', style: 'destructive', onPress: () => onRemove(person.id) },
    ]);
  };

  return (
    <View style={[styles.container, { paddingTop: insets.top + 16 }]}>
      <View style={styles.header}>
        <Text style={styles.title}>Banco de dados</Text>
        <Pressable onPress={onClose} hitSlop={12}>
          <Text style={styles.close}>Fechar</Text>
        </Pressable>
      </View>

      <FlatList
        data={people}
        keyExtractor={(item) => item.id}
        contentContainerStyle={[
          styles.list,
          people.length === 0 && styles.listEmpty,
          { paddingBottom: insets.bottom + 24 },
        ]}
        ListEmptyComponent={
          <Text style={styles.empty}>
            Nenhuma pessoa cadastrada ainda.{'\n'}Use o botao + ou capture um rosto pela camera.
          </Text>
        }
        renderItem={({ item }) => {
          const age = calculateAge(item.birthDate);
          return (
            <Pressable style={styles.row} onPress={() => onEdit(item)}>
              <Image source={{ uri: item.photoUri }} style={styles.avatar} resizeMode="cover" />
              <View style={styles.rowInfo}>
                <Text style={styles.name} numberOfLines={1}>
                  {item.name}
                </Text>
                <Text style={styles.details}>
                  {formatBirthDate(item.birthDate)}
                  {age != null ? ` · ${age} anos` : ''}
                </Text>
              </View>
              <View style={styles.rowActions}>
                <Pressable onPress={() => onEdit(item)} hitSlop={10}>
                  <Text style={styles.edit}>Editar</Text>
                </Pressable>
                <Pressable onPress={() => confirmRemove(item)} hitSlop={10}>
                  <Text style={styles.remove}>Remover</Text>
                </Pressable>
              </View>
            </Pressable>
          );
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingBottom: 16,
  },
  title: { color: colors.text, fontSize: 24, fontWeight: '700' },
  close: { color: colors.accent, fontSize: 16, fontWeight: '600' },
  list: { paddingHorizontal: 20, gap: 12 },
  listEmpty: { flexGrow: 1, justifyContent: 'center' },
  empty: { color: colors.textMuted, fontSize: 15, textAlign: 'center', lineHeight: 22 },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    padding: 12,
    borderRadius: 14,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  avatar: { width: 56, height: 56, borderRadius: 28, backgroundColor: colors.surfaceAlt },
  rowInfo: { flex: 1, gap: 4 },
  name: { color: colors.text, fontSize: 16, fontWeight: '600' },
  details: { color: colors.textMuted, fontSize: 13 },
  rowActions: { alignItems: 'flex-end', gap: 10 },
  edit: { color: colors.accent, fontSize: 14, fontWeight: '600' },
  remove: { color: colors.danger, fontSize: 14, fontWeight: '600' },
});
