import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  LayoutChangeEvent,
  PanResponder,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {
  Camera,
  runAtTargetFps,
  useCameraDevice,
  useCameraDevices,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { useFaceDetector } from 'react-native-vision-camera-face-detector';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useRunOnJS, useSharedValue } from 'react-native-worklets-core';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { FaceOverlay } from '../components/FaceOverlay';
import {
  CROP_MARGIN,
  DETECTION_FPS,
  MATCH_THRESHOLD,
  MAX_PITCH_ANGLE,
  MAX_YAW_ANGLE,
  MIN_FACE_PIXELS,
  RECHECK_EVERY_TICKS,
  REGION_SIZE,
} from '../face/constants';
import { cosineSimilarity, isValidEmbedding, l2Normalize, warpRegionToTensor } from '../face/embedding';
import {
  eyeAlignedRegion,
  orientationToUprightRotation,
  rotationToPluginValue,
  toSquareFaceBox,
  uprightFrameSize,
  uprightRectToFrameRect,
  uprightRectToViewRect,
} from '../face/geometry';
import { colors } from '../theme';
import type { FaceStatus, OverlayFace, Person } from '../types';

/** O que guardamos entre frames para nao reprocessar o mesmo rosto sem parar. */
interface CacheEntry {
  status: FaceStatus;
  label: string;
  score: number;
  tick: number;
}

interface Props {
  people: Person[];
  /** `false` enquanto alguma tela modal esta aberta, para poupar bateria. */
  isActive: boolean;
  onOpenAdd: () => void;
  onOpenList: () => void;
  /** Recebe a foto capturada pela camera para abrir o cadastro ja com ela. */
  onCaptureForEnroll: (photoUri: string) => void;
}

export function CameraScreen({
  people,
  isActive,
  onOpenAdd,
  onOpenList,
  onCaptureForEnroll,
}: Props) {
  const insets = useSafeAreaInsets();
  const { hasPermission, requestPermission } = useCameraPermission();
  const [cameraPosition, setCameraPosition] = useState<'front' | 'back'>('front');
  const camera = useRef<Camera>(null);
  const [capturing, setCapturing] = useState(false);

  // Nem todo aparelho tem as duas cameras (tablets e emuladores costumam expor
  // so uma). Pedir a frontal e desistir se nao existir deixaria a tela morta,
  // entao caimos para qualquer camera disponivel.
  const allDevices = useCameraDevices();
  const preferredDevice = useCameraDevice(cameraPosition);
  const device = preferredDevice ?? allDevices[0];
  const hasFront = allDevices.some((d) => d.position === 'front');
  const hasBack = allDevices.some((d) => d.position === 'back');
  const canFlip = hasFront && hasBack;
  const [faces, setFaces] = useState<OverlayFace[]>([]);
  const [viewSize, setViewSize] = useState({ width: 0, height: 0 });

  // --- Zoom ---------------------------------------------------------------
  // O `zoom` da VisionCamera vai de device.minZoom a device.maxZoom, com
  // `neutralZoom` sendo o "1x" (grande-angular padrao). Controlamos por estado
  // para ter tanto a pinca (PanResponder, sem lib extra) quanto os botoes.
  const minZoom = device?.minZoom ?? 1;
  const maxZoom = Math.min(device?.maxZoom ?? 1, 10); // limita a 10x: acima disso vira so ruido
  const neutralZoom = device?.neutralZoom ?? 1;
  const [zoom, setZoom] = useState(neutralZoom);

  // Ao trocar de camera, o intervalo de zoom muda; volta ao neutro.
  useEffect(() => {
    setZoom(neutralZoom);
  }, [neutralZoom, cameraPosition]);

  const clampZoom = useCallback(
    (value: number) => Math.max(minZoom, Math.min(maxZoom, value)),
    [minZoom, maxZoom]
  );

  // Distancia entre dois toques e o zoom no inicio do gesto de pinca.
  const pinchStart = useRef<{ distance: number; zoom: number } | null>(null);

  const panResponder = useMemo(
    () =>
      PanResponder.create({
        // So assume o gesto quando ha DOIS dedos, para nao roubar toques dos botoes.
        onStartShouldSetPanResponder: (e) => e.nativeEvent.touches.length === 2,
        onMoveShouldSetPanResponder: (e) => e.nativeEvent.touches.length === 2,
        onPanResponderGrant: (e) => {
          const t = e.nativeEvent.touches;
          if (t.length === 2) {
            const dx = t[0].pageX - t[1].pageX;
            const dy = t[0].pageY - t[1].pageY;
            pinchStart.current = { distance: Math.hypot(dx, dy), zoom };
          }
        },
        onPanResponderMove: (e) => {
          const t = e.nativeEvent.touches;
          if (t.length === 2 && pinchStart.current != null) {
            const dx = t[0].pageX - t[1].pageX;
            const dy = t[0].pageY - t[1].pageY;
            const distance = Math.hypot(dx, dy);
            if (pinchStart.current.distance > 0) {
              const factor = distance / pinchStart.current.distance;
              setZoom(clampZoom(pinchStart.current.zoom * factor));
            }
          }
        },
        onPanResponderRelease: () => {
          pinchStart.current = null;
        },
        onPanResponderTerminate: () => {
          pinchStart.current = null;
        },
      }),
    [zoom, clampZoom]
  );

  // Passo de zoom dos botoes: 1x do intervalo util a cada toque.
  const stepZoom = useCallback(
    (direction: 1 | -1) => setZoom((z) => clampZoom(z + direction)),
    [clampZoom]
  );

  /**
   * Interpretador EXCLUSIVO do frame processor.
   *
   * O fast-tflite guarda os buffers de saida como objetos JSI amarrados ao
   * runtime que rodou a inferencia pela primeira vez. Se a mesma instancia for
   * usada pela thread JS (cadastro) e pela thread do worklet (camera), a
   * segunda le os buffers da primeira com o ponteiro de runtime errado e o app
   * morre com SIGSEGV em `TensorflowPlugin::copyOutputBuffers`.
   *
   * Pausar a camera durante o cadastro nao resolve: sobra a janela dos frames
   * que ja estavam em voo. Um interpretador por runtime resolve de vez, ao
   * custo de carregar o modelo (5 MB) duas vezes.
   */
  const modelPlugin = useTensorflowModel(require('../../assets/models/mobilefacenet.tflite'));
  const model = modelPlugin.state === 'loaded' ? modelPlugin.model : undefined;

  const { detectFaces } = useFaceDetector({
    performanceMode: 'fast',
    // Precisamos da posicao dos olhos: e ela que define escala e rotacao do
    // alinhamento. Sem alinhar, "mesma pessoa" e "pessoas diferentes" ficam
    // com similaridades sobrepostas e nenhum limiar separa as duas.
    landmarkMode: 'all',
    contourMode: 'none',
    classificationMode: 'none',
    // Sem tracking o ML Kit nao devolve trackingId e nao daria para reaproveitar
    // o reconhecimento entre frames.
    trackingEnabled: true,
    minFaceSize: 0.15,
  });
  const { resize } = useResizePlugin();

  const recognitionCache = useSharedValue<Record<string, CacheEntry>>({});
  const tickCounter = useSharedValue(0);

  useEffect(() => {
    if (!hasPermission) void requestPermission();
  }, [hasPermission, requestPermission]);

  // Quando o banco muda, os rotulos em cache ficam desatualizados.
  useEffect(() => {
    recognitionCache.value = {};
  }, [people, recognitionCache]);

  // Sem camera ativa nao chegam frames novos, entao o overlay congelaria com as
  // ultimas caixas desenhadas.
  useEffect(() => {
    if (!isActive) {
      recognitionCache.value = {};
      setFaces([]);
    }
  }, [isActive, recognitionCache]);

  const publishFaces = useRunOnJS((next: OverlayFace[]) => {
    setFaces(next);
  }, []);


  // Somente o necessario para o matching; evita copiar fotos e datas para o
  // runtime do worklet a cada recriacao do frame processor.
  const gallery = useMemo(
    () => people.map((person) => ({ name: person.name, embedding: person.embedding })),
    [people]
  );

  // A preview so espelha quando a camera em uso e realmente a frontal.
  const mirrored = device?.position === 'front';

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (model == null || viewSize.width === 0 || viewSize.height === 0) return;

      runAtTargetFps(DETECTION_FPS, () => {
        'worklet';
        const tick = tickCounter.value + 1;
        tickCounter.value = tick;

        const rotation = orientationToUprightRotation(frame.orientation);
        const upright = uprightFrameSize(frame.width, frame.height, rotation);
        const detected = detectFaces(frame);

        const previousCache = recognitionCache.value;
        const nextCache: Record<string, CacheEntry> = {};
        const overlay: OverlayFace[] = [];

        for (let i = 0; i < detected.length; i++) {
          const face = detected[i];
          // Quando o tracking perde o id usamos o indice, com sinal negativo
          // para nunca colidir com um trackingId real.
          const key = face.trackingId != null ? face.trackingId : -(i + 1);
          const cacheKey = `${key}`;

          // O quadrado desenhado na tela continua vindo da caixa do ML Kit: e
          // ele que o usuario espera ver em volta do rosto.
          const box = toSquareFaceBox(face.bounds, CROP_MARGIN, upright.width, upright.height);
          const badAngle =
            Math.abs(face.yawAngle) > MAX_YAW_ANGLE || Math.abs(face.pitchAngle) > MAX_PITCH_ANGLE;

          const viewRect = uprightRectToViewRect(
            box != null ? box : face.bounds,
            upright.width,
            upright.height,
            viewSize.width,
            viewSize.height,
            mirrored
          );

          // Ja o recorte que vai para o modelo e posicionado pelos olhos.
          // Sem os dois olhos nao da para alinhar — normalmente e rosto de
          // perfil, que o modelo nao reconheceria bem de qualquer jeito.
          const rightEye = face.landmarks != null ? face.landmarks.RIGHT_EYE : undefined;
          const leftEye = face.landmarks != null ? face.landmarks.LEFT_EYE : undefined;
          const region =
            rightEye != null && leftEye != null
              ? eyeAlignedRegion(rightEye, leftEye, upright.width, upright.height)
              : null;

          // Rosto ilegivel: amarelo. O motivo vai no rotulo porque "nao deu"
          // sozinho nao diz ao usuario o que fazer para melhorar.
          let blockedReason: string | null = null;
          if (box == null) blockedReason = 'Rosto fora do quadro';
          else if (box.width < MIN_FACE_PIXELS) blockedReason = 'Aproxime-se mais';
          else if (rightEye == null || leftEye == null) blockedReason = 'Olhos nao detectados';
          else if (badAngle) blockedReason = 'Olhe para a camera';
          else if (region == null) blockedReason = 'Rosto fora do quadro';

          // As checagens de null aparecem de novo aqui porque o TypeScript nao
          // estreita os tipos atraves da variavel `blockedReason`.
          if (
            blockedReason != null ||
            box == null ||
            region == null ||
            rightEye == null ||
            leftEye == null
          ) {
            overlay.push({
              key,
              x: viewRect.x,
              y: viewRect.y,
              width: viewRect.width,
              height: viewRect.height,
              status: 'unreadable',
              label: blockedReason ?? 'Nao foi possivel ler',
              score: 0,
            });
            continue;
          }

          const cached = previousCache[cacheKey];
          const isFresh =
            cached != null && cached.status !== 'unreadable' && tick - cached.tick < RECHECK_EVERY_TICKS;

          let entry: CacheEntry;
          if (isFresh) {
            entry = cached;
          } else {
            entry = { status: 'unreadable', label: 'Falha ao processar', score: 0, tick };
            try {
              // Regiao maior que o rosto, posicionada e escalada pelos olhos.
              const cropRect = uprightRectToFrameRect(region, rotation, frame.width, frame.height);
              const rgb = resize(frame, {
                crop: cropRect,
                scale: { width: REGION_SIZE, height: REGION_SIZE },
                // O recorte sai no espaco do buffer bruto; girar aqui deixa a
                // regiao em pe, que e o espaco em que os olhos foram medidos.
                rotation: rotationToPluginValue(rotation),
                pixelFormat: 'rgb',
                dataType: 'uint8',
              });

              // Olhos em coordenadas do buffer buscado.
              const toRegion = REGION_SIZE / region.width;
              const tensor = warpRegionToTensor(
                rgb,
                REGION_SIZE,
                (rightEye.x - region.x) * toRegion,
                (rightEye.y - region.y) * toRegion,
                (leftEye.x - region.x) * toRegion,
                (leftEye.y - region.y) * toRegion
              );
              if (tensor == null) throw new Error('warp falhou');

              const output = model.runSync([tensor]);
              const raw = output[0];

              if (isValidEmbedding(raw)) {
                const embedding = l2Normalize(raw);

                let bestScore = -1;
                let bestName = '';
                for (let p = 0; p < gallery.length; p++) {
                  const score = cosineSimilarity(embedding, gallery[p].embedding);
                  if (score > bestScore) {
                    bestScore = score;
                    bestName = gallery[p].name;
                  }
                }

                if (bestScore >= MATCH_THRESHOLD) {
                  entry = { status: 'known', label: bestName, score: bestScore, tick };
                } else {
                  entry = {
                    status: 'unknown',
                    label: 'Desconhecido',
                    score: bestScore < 0 ? 0 : bestScore,
                    tick,
                  };
                }
              }
            } catch {
              // Recorte invalido ou falha de inferencia: mantem o amarelo com o
              // rotulo generico definido acima.
            }
          }

          nextCache[cacheKey] = entry;
          overlay.push({
            key,
            x: viewRect.x,
            y: viewRect.y,
            width: viewRect.width,
            height: viewRect.height,
            status: entry.status,
            label: entry.label,
            score: entry.score,
          });
        }

        recognitionCache.value = nextCache;
        publishFaces(overlay);
      });
    },
    [model, gallery, viewSize, mirrored, detectFaces, resize, publishFaces]
  );

  const handleLayout = useCallback((event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    setViewSize({ width, height });
  }, []);

  // Tira uma foto nitida e entrega ao cadastro. Usamos takePhoto (alta
  // resolucao) em vez de um frame do preview: o mesmo pipeline de imagem
  // estatica do cadastro por galeria roda sobre ela, com deteccao "accurate".
  const handleCapture = useCallback(async () => {
    if (camera.current == null || capturing) return;
    setCapturing(true);
    try {
      const photo = await camera.current.takePhoto({ enableShutterSound: false });
      // takePhoto devolve um caminho de arquivo sem esquema; o resto do app usa URIs.
      const uri = photo.path.startsWith('file://') ? photo.path : `file://${photo.path}`;
      onCaptureForEnroll(uri);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      Alert.alert('Nao foi possivel capturar', message);
    } finally {
      setCapturing(false);
    }
  }, [capturing, onCaptureForEnroll]);

  if (!hasPermission) {
    return (
      <Centered>
        <Text style={styles.message}>O app precisa de acesso a camera.</Text>
        <Pressable style={styles.primaryButton} onPress={() => void requestPermission()}>
          <Text style={styles.primaryButtonText}>Permitir camera</Text>
        </Pressable>
      </Centered>
    );
  }

  if (device == null) {
    // A lista comeca vazia enquanto a VisionCamera enumera os dispositivos;
    // so e "sem camera" de verdade depois que a enumeracao terminou vazia.
    return (
      <Centered>
        <ActivityIndicator size="large" color={colors.accent} />
        <Text style={styles.message}>
          {allDevices.length === 0
            ? 'Procurando camera...'
            : 'Nenhuma camera compativel neste aparelho.'}
        </Text>
      </Centered>
    );
  }

  const knownCount = faces.filter((f) => f.status === 'known').length;
  const unknownCount = faces.filter((f) => f.status === 'unknown').length;
  const zoomLabel = `${(zoom / neutralZoom).toFixed(1)}x`;
  const canZoom = maxZoom - minZoom > 0.1;

  return (
    <View style={styles.container} onLayout={handleLayout} {...panResponder.panHandlers}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isActive}
        frameProcessor={frameProcessor}
        zoom={zoom}
        photo
        // O overlay assume aspect-fill; mudar isso quebraria o alinhamento.
        resizeMode="cover"
        pixelFormat="yuv"
      />

      <FaceOverlay faces={faces} />

      <View style={[styles.topBar, { paddingTop: insets.top + 12 }]}>
        <Pressable
          style={[styles.circleButton, styles.addButton]}
          onPress={onOpenAdd}
          accessibilityLabel="Cadastrar nova pessoa"
        >
          <Text style={styles.addButtonText}>+</Text>
        </Pressable>

        <View style={styles.statusPill}>
          {modelPlugin.state === 'loading' ? (
            <ActivityIndicator size="small" color={colors.text} />
          ) : (
            <Text style={styles.statusText} numberOfLines={2}>
              {faces.length} rosto{faces.length === 1 ? '' : 's'}
              {knownCount > 0 ? ` · ${knownCount} reconhecido${knownCount === 1 ? '' : 's'}` : ''}
            </Text>
          )}
        </View>

        {canFlip ? (
          <Pressable
            style={styles.circleButton}
            onPress={() => setCameraPosition((p) => (p === 'front' ? 'back' : 'front'))}
            accessibilityLabel="Trocar camera"
          >
            <Text style={styles.iconText}>⟲</Text>
          </Pressable>
        ) : (
          // Mantem o espaco para o titulo do meio nao sair do centro.
          <View style={styles.circleButtonPlaceholder} />
        )}
      </View>

      {canZoom && (
        <View style={[styles.zoomColumn, { top: insets.top + 96 }]}>
          <Pressable style={styles.zoomButton} onPress={() => stepZoom(1)} hitSlop={8}>
            <Text style={styles.zoomButtonText}>+</Text>
          </Pressable>
          <Pressable style={styles.zoomPill} onPress={() => setZoom(neutralZoom)} hitSlop={8}>
            <Text style={styles.zoomPillText}>{zoomLabel}</Text>
          </Pressable>
          <Pressable style={styles.zoomButton} onPress={() => stepZoom(-1)} hitSlop={8}>
            <Text style={styles.zoomButtonText}>−</Text>
          </Pressable>
        </View>
      )}

      <View style={[styles.bottomBar, { paddingBottom: insets.bottom + 16 }]}>
        <View style={styles.legend}>
          <LegendItem color={colors.known} label="Cadastrado" />
          <LegendItem color={colors.unknown} label="Desconhecido" />
          <LegendItem color={colors.unreadable} label="Ilegivel" />
        </View>

        <View style={styles.actionRow}>
          <Pressable style={styles.listButton} onPress={onOpenList}>
            <Text style={styles.listButtonText}>Banco ({people.length})</Text>
          </Pressable>

          <Pressable
            style={[styles.captureButton, capturing && styles.buttonDisabled]}
            onPress={() => void handleCapture()}
            disabled={capturing}
            accessibilityLabel="Capturar rosto para cadastrar"
          >
            {capturing ? (
              <ActivityIndicator color={colors.background} />
            ) : (
              <View style={styles.captureInner} />
            )}
          </Pressable>

          <View style={styles.listButtonPlaceholder} />
        </View>

        <Text style={styles.captureHint}>
          {unknownCount > 0
            ? 'Rosto desconhecido — toque no circulo para cadastrar'
            : 'Toque no circulo para cadastrar pela camera'}
        </Text>
      </View>

      {modelPlugin.state === 'error' && (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>
            Falha ao carregar o modelo: {modelPlugin.error.message}
          </Text>
        </View>
      )}
    </View>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <View style={styles.legendItem}>
      <View style={[styles.legendDot, { backgroundColor: color }]} />
      <Text style={styles.legendLabel}>{label}</Text>
    </View>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <View style={styles.centered}>{children}</View>;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    gap: 20,
    backgroundColor: colors.background,
  },
  message: { color: colors.text, fontSize: 16, textAlign: 'center' },
  topBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingBottom: 12,
    gap: 12,
  },
  circleButton: {
    width: 52,
    height: 52,
    borderRadius: 26,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(21, 30, 49, 0.85)',
    borderWidth: 1,
    borderColor: colors.border,
  },
  circleButtonPlaceholder: { width: 52, height: 52 },
  addButton: { backgroundColor: colors.accent, borderColor: colors.accent },
  addButtonText: {
    color: colors.background,
    fontSize: 32,
    fontWeight: '700',
    lineHeight: 36,
    marginTop: -2,
  },
  iconText: { color: colors.text, fontSize: 22 },
  statusPill: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 20,
    backgroundColor: 'rgba(11, 17, 32, 0.75)',
  },
  statusText: { color: colors.text, fontSize: 13, fontWeight: '600' },
  bottomBar: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    paddingHorizontal: 16,
    paddingTop: 16,
    gap: 14,
    backgroundColor: 'rgba(11, 17, 32, 0.75)',
  },
  legend: { flexDirection: 'row', justifyContent: 'center', gap: 18 },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  legendDot: { width: 12, height: 12, borderRadius: 6 },
  legendLabel: { color: colors.text, fontSize: 12 },
  actionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  listButton: {
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  listButtonText: { color: colors.text, fontSize: 14, fontWeight: '600' },
  // Reserva a mesma largura do botao "Banco" do outro lado para centralizar
  // o botao de captura sem medir texto.
  listButtonPlaceholder: { width: 96 },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.accent,
    borderWidth: 4,
    borderColor: 'rgba(255,255,255,0.35)',
  },
  captureInner: {
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: colors.background,
  },
  captureHint: { color: colors.textMuted, fontSize: 12, textAlign: 'center' },
  zoomColumn: {
    position: 'absolute',
    right: 16,
    alignItems: 'center',
    gap: 10,
  },
  zoomButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(11, 17, 32, 0.75)',
    borderWidth: 1,
    borderColor: colors.border,
  },
  zoomButtonText: { color: colors.text, fontSize: 24, fontWeight: '600', lineHeight: 26 },
  zoomPill: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: 'rgba(11, 17, 32, 0.75)',
  },
  zoomPillText: { color: colors.text, fontSize: 12, fontWeight: '700' },
  primaryButton: {
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 12,
    backgroundColor: colors.accent,
  },
  primaryButtonText: { color: colors.background, fontWeight: '700', fontSize: 16 },
  buttonDisabled: { opacity: 0.6 },
  errorBanner: {
    position: 'absolute',
    left: 16,
    right: 16,
    bottom: 140,
    padding: 12,
    borderRadius: 10,
    backgroundColor: colors.danger,
  },
  errorText: { color: colors.background, fontSize: 13, fontWeight: '600' },
});
