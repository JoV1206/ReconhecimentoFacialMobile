import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { statusColor } from '../theme';
import type { OverlayFace } from '../types';

interface Props {
  faces: OverlayFace[];
}

/**
 * Desenha o quadrado de rastreamento sobre a preview da camera e a etiqueta
 * com o nome logo abaixo dele.
 */
function FaceOverlayComponent({ faces }: Props) {
  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {faces.map((face) => {
        const color = statusColor(face.status);
        return (
          <View
            key={face.key}
            style={[
              styles.box,
              {
                left: face.x,
                top: face.y,
                width: face.width,
                height: face.height,
                borderColor: color,
              },
            ]}
          >
            <View style={[styles.labelWrapper, { backgroundColor: color }]}>
              <Text style={styles.label} numberOfLines={1}>
                {face.label}
              </Text>
            </View>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    position: 'absolute',
    borderWidth: 3,
    borderRadius: 10,
  },
  labelWrapper: {
    position: 'absolute',
    // Encosta a etiqueta na borda de baixo do quadrado.
    top: '100%',
    marginTop: 6,
    alignSelf: 'center',
    maxWidth: 240,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 6,
  },
  label: {
    color: '#0B1120',
    fontSize: 14,
    fontWeight: '700',
  },
});

export const FaceOverlay = React.memo(FaceOverlayComponent);
