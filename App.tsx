import React, { useCallback, useEffect, useState } from 'react';
import { Modal, StyleSheet, View } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { useTensorflowModel } from 'react-native-fast-tflite';

import { addPerson, loadPeople, removePerson, updatePerson } from './src/db/people';
import { AddPersonScreen } from './src/screens/AddPersonScreen';
import { CameraScreen } from './src/screens/CameraScreen';
import { PeopleScreen } from './src/screens/PeopleScreen';
import { colors } from './src/theme';
import type { Person } from './src/types';

/** O que a tela de cadastro/edicao esta fazendo neste momento. */
type Editor =
  | { mode: 'closed' }
  | { mode: 'create' }
  | { mode: 'create'; photoUri: string } // foto capturada pela camera
  | { mode: 'edit'; person: Person };

export default function App() {
  const [people, setPeople] = useState<Person[]>([]);
  const [editor, setEditor] = useState<Editor>({ mode: 'closed' });
  const [showList, setShowList] = useState(false);

  /**
   * Interpretador do CADASTRO (thread JS). A camera carrega o seu proprio, de
   * proposito: o fast-tflite amarra os buffers de saida ao runtime que rodou a
   * inferencia primeiro, e compartilhar uma instancia entre a thread JS e a do
   * worklet derruba o app com SIGSEGV. Veja o comentario em CameraScreen.
   */
  const modelPlugin = useTensorflowModel(require('./assets/models/mobilefacenet.tflite'));
  const model = modelPlugin.state === 'loaded' ? modelPlugin.model : undefined;

  useEffect(() => {
    void loadPeople().then(setPeople);
  }, []);

  const editorOpen = editor.mode !== 'closed';

  const handleSaved = useCallback(
    async (person: Person) => {
      // Se o id ja existe no banco, e edicao; senao, cadastro novo.
      const exists = people.some((p) => p.id === person.id);
      setPeople(exists ? await updatePerson(person) : await addPerson(person));
      setEditor({ mode: 'closed' });
    },
    [people]
  );

  const handleRemove = useCallback(async (id: string) => {
    setPeople(await removePerson(id));
  }, []);

  return (
    <SafeAreaProvider>
      <View style={styles.root}>
        <StatusBar style="light" />

        <CameraScreen
          people={people}
          isActive={!editorOpen && !showList}
          onOpenAdd={() => setEditor({ mode: 'create' })}
          onOpenList={() => setShowList(true)}
          onCaptureForEnroll={(photoUri) => setEditor({ mode: 'create', photoUri })}
        />

        <Modal
          visible={editorOpen}
          animationType="slide"
          presentationStyle="fullScreen"
          onRequestClose={() => setEditor({ mode: 'closed' })}
        >
          <AddPersonScreen
            model={model}
            person={editor.mode === 'edit' ? editor.person : undefined}
            initialPhotoUri={
              editor.mode === 'create' && 'photoUri' in editor ? editor.photoUri : undefined
            }
            onCancel={() => setEditor({ mode: 'closed' })}
            onSaved={(person) => void handleSaved(person)}
          />
        </Modal>

        <Modal
          visible={showList}
          animationType="slide"
          presentationStyle="fullScreen"
          onRequestClose={() => setShowList(false)}
        >
          <PeopleScreen
            people={people}
            onClose={() => setShowList(false)}
            onRemove={(id) => void handleRemove(id)}
            onEdit={(person) => {
              setShowList(false);
              setEditor({ mode: 'edit', person });
            }}
          />
        </Modal>
      </View>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.background },
});
