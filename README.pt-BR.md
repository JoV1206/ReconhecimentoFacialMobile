# Reconhecimento Facial em Tempo Real

[English](README.md) | **Português**

![Platform](https://img.shields.io/badge/platform-Android-3DDC84)
![Expo SDK](https://img.shields.io/badge/Expo%20SDK-54-000020)
![React Native](https://img.shields.io/badge/React%20Native-0.81-61DAFB)
![License](https://img.shields.io/badge/license-MIT-blue)

Aplicativo Android em React Native e **Expo SDK 54** que, pela câmera, detecta e rastreia
**todos os rostos** em cena em tempo real e identifica quem já está cadastrado. Todo o
processamento — detecção, alinhamento e reconhecimento — roda **no próprio aparelho**;
nenhuma imagem sai do celular.

## Índice

- [Visão geral](#visão-geral)
- [Funcionalidades](#funcionalidades)
- [Como funciona](#como-funciona)
- [Notas de implementação](#notas-de-implementação)
- [Requisitos](#requisitos)
- [Primeiros passos](#primeiros-passos)
- [Gerando um APK](#gerando-um-apk)
- [Build na nuvem e atualizações OTA](#build-na-nuvem-e-atualizações-ota)
- [Testando sem aparelho](#testando-sem-aparelho)
- [Como usar](#como-usar)
- [Configuração](#configuração)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Privacidade](#privacidade)
- [Créditos](#créditos)
- [Licença](#licença)

## Visão geral

Cada rosto rastreado é desenhado com um quadrado colorido e uma etiqueta:

| Cor do quadrado | Significado |
| --- | --- |
| **Verde** | Pessoa cadastrada — o nome aparece abaixo do quadrado |
| **Vermelho** | Rosto lido normalmente, mas sem correspondência no banco (`Desconhecido`) |
| **Amarelo** | Rosto detectado, porém ilegível (pequeno demais, muito de lado, sem os dois olhos ou fora do quadro) — a etiqueta indica o motivo |

## Funcionalidades

- **Detecção e rastreamento de múltiplos rostos** ao vivo, com quadrado colorido e
  etiqueta por rosto.
- **Reconhecimento facial no aparelho** contra um banco local, com embeddings do
  MobileFaceNet.
- **Três formas de cadastrar** uma pessoa:
  - botão **`+`** (topo esquerdo) — escolher da galeria ou tirar uma foto;
  - **captura pela câmera** — o botão de captura (círculo) tira a foto da cena atual e já
    abre o formulário de cadastro com ela, ideal para cadastrar um desconhecido na hora;
  - ambas pedem **nome** e **data de nascimento**.
- **Editar** um cadastro (nome, data e foto) pela lista do banco. Trocar só o texto é
  instantâneo; trocar a foto reprocessa o rosto.
- **Zoom** por pinça (dois dedos) ou pelos botões **+ / −**, com indicador de nível.
- **Banco de dados local** no aparelho (`AsyncStorage` mais fotos no diretório do app),
  com listagem, edição e remoção.
- **Atualização OTA** via EAS Update para correções de JavaScript sem recompilar.

## Como funciona

```
Câmera (VisionCamera, frame processor a 8 fps)
   |- ML Kit Face Detection ......... caixas + posição dos olhos + trackingId
        |- região 192x192 em volta do rosto (vision-camera-resize-plugin, nativo)
             |- alinhamento pelos olhos -> 112x112 (warp afim no worklet)
                  |- MobileFaceNet .tflite (react-native-fast-tflite) -> 192 dims
                       |- cosseno contra o banco local -> verde / vermelho / amarelo
```

**Modelo:** [MobileFaceNet](https://arxiv.org/abs/1804.07573) em TensorFlow Lite
(`assets/models/mobilefacenet.tflite`, 5,2 MB), obtido do projeto
[estebanuri/face_recognition](https://github.com/estebanuri/face_recognition).
Entrada `[1, 112, 112, 3]` float32, saída `[1, 192]`.

## Notas de implementação

- **Alinhamento pelos olhos, não recorte pela caixa.** O recorte que vai ao modelo é
  posicionado e escalado pela distância entre os olhos, levando-os às posições do gabarito
  ArcFace com que o MobileFaceNet foi treinado. Isso **não é acabamento**: medido em 253
  pares de rostos reais, recortar pela caixa do detector deixa as similaridades de "mesma
  pessoa" e "pessoas diferentes" sobrepostas — nenhum limiar separa as duas. Com
  alinhamento, a separação vira uma folga de cerca de 0,34. Detalhes em
  [scripts/offline-eval](scripts/offline-eval/README.md).
- **Warp em dois estágios.** O resize plugin só recorta, escala, gira em múltiplos de 90
  graus e espelha — não faz warp afim. Então o lado nativo entrega uma região 192x192 e o
  warp de alinhamento final acontece em JS dentro do worklet (~1,4 ms estimado; o
  orçamento por ciclo é de 125 ms).
- **Reaproveitamento por `trackingId`.** O ML Kit mantém um id por rosto entre frames,
  então a inferência só roda de novo a cada ~2 s por pessoa. Sem isso, 4 rostos a 8 fps
  seriam 32 inferências por segundo.
- **Rotação.** O `frame.orientation` da VisionCamera é o inverso do `rotationDegrees` que
  o ML Kit recebe. `src/face/geometry.ts` desfaz essa inversão para mapear as caixas (que
  chegam no espaço "em pé") de volta ao buffer bruto, onde o resize plugin recorta. O
  recorte usa origem e lado pares por causa da subamostragem de cor do YUV 4:2:0.
- **Um interpretador TFLite por thread.** O `react-native-fast-tflite` amarra os buffers
  de saída ao runtime JSI que rodou a primeira inferência. Compartilhar uma única
  instância entre a thread JS (cadastro) e a do worklet (câmera) derruba o app com
  `SIGSEGV` em `copyOutputBuffers`. A solução é **cada runtime carregar seu próprio
  modelo** (`App.tsx` para o cadastro, `CameraScreen.tsx` para a câmera).
- **`isFinite` e globais afins não existem em worklet.** O runtime paralelo do
  worklets-core não compartilha funções globais do JS; chamá-las dentro de um `'worklet'`
  lança em tempo de execução, só no aparelho. Uma varredura (`npm run verify:worklets`)
  guarda contra isso.
- **Cadastro por foto assa o EXIF.** O detector de imagem estática usa rotação 0 e ignora
  o EXIF, então a foto passa antes pelo `expo-image-manipulator`, que grava a rotação nos
  pixels e remove o EXIF, mantendo as coordenadas dos olhos consistentes com o arquivo.

## Requisitos

- **Android 8.0 (API 26) ou superior.** O detector de rostos do ML Kit exige
  `minSdkVersion 26`, acima do padrão 24 do SDK 54 — por isso o `expo-build-properties` no
  `app.json` sobe esse valor. Sem isso o build falha na fusão do manifesto.
- Node 20+ e JDK 17.
- Android SDK (via Android Studio) com `ANDROID_HOME` configurado.
- De preferência um **aparelho Android físico** — a câmera do emulador é sintética (dá
  para apontar para a webcam do PC, veja abaixo).

> **Não funciona no Expo Go.** O Expo Go é um app pronto com um conjunto fixo de módulos
> nativos, e VisionCamera, TensorFlow Lite, ML Kit e worklets não estão entre eles. Abrir
> o projeto no Expo Go carrega o JavaScript mas falha ao resolver esses módulos — dá tela
> branca ou erro de módulo ausente. Não há configuração que resolva: é preciso um **build
> nativo próprio** (`npx expo run:android`) ou o APK instalado.

## Primeiros passos

```bash
npm install

# compila e instala em um aparelho conectado (depuração USB ativada)
npx expo run:android

# depois, no dia a dia, basta o Metro:
npm start
```

## Gerando um APK

```bash
cd android
./gradlew assembleRelease -PreactNativeArchitectures=arm64-v8a
# APK em: android/app/build/outputs/apk/release/app-release.apk
```

Remova `-PreactNativeArchitectures=arm64-v8a` para gerar um APK universal (todas as ABIs),
que demora bem mais para compilar. `arm64-v8a` cobre praticamente qualquer celular Android
atual — mas o **emulador precisa de `x86_64`**:

```bash
./gradlew assembleRelease -PreactNativeArchitectures=x86_64
```

No emulador, configure a câmera frontal para a webcam do computador, senão o app abre mas
não vê rosto nenhum. Em `~/.android/avd/<nome>.avd/config.ini`:

```ini
hw.camera.front=webcam0
hw.camera.back=virtualscene
```

Deixar as duas como `webcam0` **não** funciona: só uma câmera é criada, e o CameraX se
recusa a inicializar quando falta a outra (`CameraIdListIncorrectException`).

> Esse APK é assinado com a **chave de debug**: instala e roda em qualquer aparelho, mas
> não serve para a Play Store. Para publicar, gere uma chave de upload própria e configure
> `signingConfigs.release`, ou use `eas build`.

## Build na nuvem e atualizações OTA

```bash
# build na nuvem, sem precisar do SDK local
npx eas build --platform android --profile preview

# atualização OTA (só JavaScript): correções de lógica/texto sem recompilar
npx eas update --channel production --message "descrição da mudança"
```

Os canais (`development` / `preview` / `production`) estão em `eas.json`. O OTA atualiza
**apenas JavaScript**; mudança nativa, ícone ou permissão exige um APK novo.

## Testando sem aparelho

```bash
npm run verify            # roda os quatro abaixo em sequência

npm run typecheck         # tsc --noEmit
npm run verify:math       # rotação de coordenadas e similaridade de cosseno
npm run verify:alignment  # warp de alinhamento vs. implementação de referência
npm run verify:worklets   # globais proibidos dentro de worklets
```

Esses testes existem porque os erros dessa parte não quebram nada visivelmente: se a
conversão de coordenadas ou o warp estiverem errados, o recorte pega o pedaço errado do
frame e o reconhecimento vira ruído silenciosamente.

- `verify:math` — ida-e-volta das 4 orientações contra um modelo de rotação independente,
  mais os casos de borda do recorte.
- `verify:alignment` — carrega os arquivos `.ts` de `src/face/` e confere que produzem o
  **mesmo tensor 112x112**, número a número, que a implementação Python de referência.
  Depende de fixtures locais geradas por
  [scripts/offline-eval](scripts/offline-eval/README.md) (não versionadas, pois derivam de
  fotos reais); sem elas o teste é pulado.
- `verify:worklets` — procura funções globais do JS dentro de blocos `'worklet'`, que o
  worklets-core não consegue compartilhar (falha só no aparelho).

## Como usar

1. Abra o app e conceda a permissão de câmera.
2. **Cadastre** de uma destas formas:
   - toque no **`+`** (topo esquerdo) e escolha uma foto da galeria ou tire uma na hora;
   - **ou** aponte a câmera para a pessoa e toque no **círculo de captura** (barra de
     baixo) — a foto da cena vira o cadastro.
   - Use um rosto nítido, de frente e com **apenas a pessoa** a ser cadastrada.
3. Preencha **nome** e **data de nascimento** (`DD/MM/AAAA`) e salve.
4. Volte para a câmera: o rosto passa a sair com quadrado **verde** e o nome.
5. **Zoom:** pinça com dois dedos ou os botões **+ / −** na lateral; toque no nível (por
   exemplo, `1.0x`) para voltar ao normal.
6. **Banco de dados** (barra de baixo): lista os cadastros. Toque numa pessoa para
   **editar** nome, data ou foto, ou use **Remover**.

## Configuração

Tudo em [`src/face/constants.ts`](src/face/constants.ts):

| Constante | Padrão | Para que serve |
| --- | --- | --- |
| `MATCH_THRESHOLD` | `0.62` | Cosseno mínimo para dizer "é a mesma pessoa". Calibrado: nos testes, mesma pessoa ficou >= 0,784 e pessoas diferentes <= 0,448 — qualquer valor entre ~0,50 e ~0,75 acerta tudo. |
| `DETECTION_FPS` | `8` | Frequência de detecção. Menor = mais bateria e menos calor. |
| `RECHECK_EVERY_TICKS` | `16` | De quantos em quantos ciclos reconfirmar um rosto já identificado (16 é ~2 s). |
| `MIN_FACE_PIXELS` | `64` | Abaixo disso o rosto vira amarelo em vez de ser reconhecido. |
| `MAX_YAW_ANGLE` / `MAX_PITCH_ANGLE` | `45` / `40` | Quanto o rosto pode estar virado antes de virar amarelo. |

Uma foto por pessoa já funciona bem — nos testes, a mesma pessoa fotografada em dias,
luzes e câmeras diferentes deu cerca de 0,86 de similaridade. Para robustez extra, o
caminho natural é guardar **várias** fotos por pessoa e comparar contra o melhor
resultado.

> Os embeddings salvos só são comparáveis se vierem do mesmo pré-processamento. Por isso a
> chave do banco carrega uma versão (`.../people/v2-aligned`): quando o recorte muda, os
> cadastros antigos são descartados em vez de degradar o reconhecimento em silêncio.

## Estrutura do projeto

```
App.tsx                        estado global (banco + telas) e modelo do cadastro
src/
  db/people.ts                 CRUD em AsyncStorage, idade e formatação de data
  face/
    constants.ts               limiares e parâmetros ajustáveis
    embedding.ts               normalização, warp de alinhamento, L2 e cosseno (worklets)
    geometry.ts                rotação frame <-> tela e região alinhada (worklets)
    enroll.ts                  cadastro por foto: imagem -> rosto -> embedding
  screens/
    CameraScreen.tsx           câmera, zoom, captura, frame processor e reconhecimento
    AddPersonScreen.tsx        formulário de cadastro/edição
    PeopleScreen.tsx           lista do banco (editar / remover)
  components/FaceOverlay.tsx   quadrados coloridos e etiquetas
assets/models/mobilefacenet.tflite
scripts/
  generate-icons.py            gera os ícones do app por código
  verify-*.js                  testes que rodam sem aparelho (ver acima)
  offline-eval/                avaliação de qualidade em Python (ver README próprio)
```

## Privacidade

Fotos e vetores faciais ficam **apenas** no armazenamento interno do aplicativo; não há
envio para servidor algum. Dado biométrico é dado pessoal sensível: se for usar com outras
pessoas, colete o consentimento delas. As imagens de teste usadas no desenvolvimento
**não** fazem parte deste repositório, justamente por serem rostos de pessoas reais.

## Créditos

- Modelo MobileFaceNet (TFLite) de
  [estebanuri/face_recognition](https://github.com/estebanuri/face_recognition).
- Detecção de rostos: ML Kit via
  [react-native-vision-camera-face-detector](https://github.com/luicfrr/react-native-vision-camera-face-detector).
- Câmera e frame processors:
  [react-native-vision-camera](https://github.com/mrousavy/react-native-vision-camera) e
  [react-native-fast-tflite](https://github.com/mrousavy/react-native-fast-tflite).

## Licença

Distribuído sob a [Licença MIT](LICENSE).
