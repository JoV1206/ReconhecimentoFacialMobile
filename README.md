# Real-Time Face Recognition

**English** | [Português](README.pt-BR.md)

![Platform](https://img.shields.io/badge/platform-Android-3DDC84)
![Expo SDK](https://img.shields.io/badge/Expo%20SDK-54-000020)
![React Native](https://img.shields.io/badge/React%20Native-0.81-61DAFB)
![License](https://img.shields.io/badge/license-MIT-blue)

An Android app built with React Native and **Expo SDK 54** that uses the camera to
detect and track **every face** in view in real time and identify people who have been
enrolled. All processing — detection, alignment, and recognition — runs **on the
device**; no image ever leaves the phone.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Implementation Notes](#implementation-notes)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Building an APK](#building-an-apk)
- [Cloud Builds and OTA Updates](#cloud-builds-and-ota-updates)
- [Testing Without a Device](#testing-without-a-device)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Privacy](#privacy)
- [Credits](#credits)
- [License](#license)

## Overview

Each tracked face is drawn with a colored box and a label:

| Box color | Meaning |
| --- | --- |
| **Green** | Enrolled person — the name is shown below the box |
| **Red** | Face read normally but with no match in the database (`Desconhecido` / Unknown) |
| **Yellow** | Face detected but unreadable (too small, too angled, eyes not found, or out of frame) — the label states the reason |

## Features

- **Multi-face detection and tracking** in real time, with a colored box and label per
  face.
- **On-device face recognition** against a local database, using MobileFaceNet
  embeddings.
- **Three ways to enroll** a person:
  - the **`+`** button (top left) — pick from the gallery or take a photo;
  - **capture from the camera** — the capture (circle) button takes a photo of the
    current scene and opens the enrollment form with it, ideal for enrolling an unknown
    person on the spot;
  - both require a **name** and a **date of birth**.
- **Edit** an enrollment (name, date, and photo) from the database list. Changing only
  text is instant; changing the photo reprocesses the face.
- **Zoom** via pinch (two fingers) or the **+ / −** buttons, with a level indicator.
- **Local database** on the device (`AsyncStorage` plus photos in the app directory),
  with listing, editing, and removal.
- **Over-the-air (OTA) updates** through EAS Update for JavaScript fixes without
  rebuilding.

## How It Works

```
Camera (VisionCamera, frame processor at 8 fps)
   |- ML Kit Face Detection ......... boxes + eye positions + trackingId
        |- 192x192 region around the face (vision-camera-resize-plugin, native)
             |- eye-based alignment -> 112x112 (affine warp in a worklet)
                  |- MobileFaceNet .tflite (react-native-fast-tflite) -> 192 dims
                       |- cosine similarity against the local database -> green / red / yellow
```

**Model:** [MobileFaceNet](https://arxiv.org/abs/1804.07573) in TensorFlow Lite
(`assets/models/mobilefacenet.tflite`, 5.2 MB), taken from
[estebanuri/face_recognition](https://github.com/estebanuri/face_recognition).
Input `[1, 112, 112, 3]` float32, output `[1, 192]`.

## Implementation Notes

- **Eye-based alignment, not box cropping.** The crop fed to the model is positioned and
  scaled by the distance between the eyes, moving them to the ArcFace template positions
  the MobileFaceNet was trained on. This is **not polish**: measured over 253 pairs of
  real faces, cropping by the detector box leaves the "same person" and "different
  people" similarity distributions overlapping — no threshold separates them. With
  alignment the gap becomes about 0.34. See
  [scripts/offline-eval](scripts/offline-eval/README.md).
- **Two-stage warp.** The resize plugin only crops, scales, rotates by multiples of 90
  degrees, and mirrors — it does not do an affine warp. So the native side delivers a
  192x192 region and the final alignment warp runs in JS inside the worklet
  (~1.4 ms estimated; the per-cycle budget is 125 ms).
- **Reuse by `trackingId`.** ML Kit keeps an id per face across frames, so inference only
  reruns every ~2 s per person. Without it, 4 faces at 8 fps would be 32 inferences per
  second.
- **Rotation.** VisionCamera's `frame.orientation` is the inverse of the
  `rotationDegrees` ML Kit receives. `src/face/geometry.ts` undoes that inversion to map
  boxes (which arrive in the "upright" space) back to the raw buffer, where the resize
  plugin crops. The crop uses even origin and side because of YUV 4:2:0 chroma
  subsampling.
- **One TFLite interpreter per thread.** `react-native-fast-tflite` binds output buffers
  to the JSI runtime that ran the first inference. Sharing a single instance between the
  JS thread (enrollment) and the worklet thread (camera) crashes the app with `SIGSEGV`
  in `copyOutputBuffers`. The fix is for **each runtime to load its own model**
  (`App.tsx` for enrollment, `CameraScreen.tsx` for the camera).
- **`isFinite` and similar globals do not exist in a worklet.** The worklets-core
  parallel runtime cannot share plain JS global functions; calling one inside a
  `'worklet'` throws at runtime, only on the device. A scan (`npm run verify:worklets`)
  guards against it.
- **Photo enrollment bakes the EXIF.** The static-image detector uses rotation 0 and
  ignores EXIF, so the photo first passes through `expo-image-manipulator`, which bakes
  the rotation into the pixels and strips EXIF, keeping eye coordinates consistent with
  the file.

## Requirements

- **Android 8.0 (API 26) or newer.** ML Kit face detection requires `minSdkVersion 26`,
  above the SDK 54 default of 24 — this is why `expo-build-properties` in `app.json`
  raises it. Without it the build fails at manifest merge.
- Node 20+ and JDK 17.
- Android SDK (via Android Studio) with `ANDROID_HOME` configured.
- Preferably a **physical Android device** — the emulator camera is synthetic (you can
  point it at the PC webcam, see below).

> **Does not work in Expo Go.** Expo Go is a prebuilt app with a fixed set of native
> modules, and VisionCamera, TensorFlow Lite, ML Kit, and worklets are not among them.
> Opening the project in Expo Go loads the JavaScript but fails to resolve those modules
> — you get a blank screen or a missing-module error. There is no configuration that
> fixes this: you need a **native build of your own** (`npx expo run:android`) or the
> installed APK.

## Getting Started

```bash
npm install

# build and install on a connected device (USB debugging enabled)
npx expo run:android

# afterwards, day to day, just run Metro:
npm start
```

## Building an APK

```bash
cd android
./gradlew assembleRelease -PreactNativeArchitectures=arm64-v8a
# APK at: android/app/build/outputs/apk/release/app-release.apk
```

Drop `-PreactNativeArchitectures=arm64-v8a` to produce a universal APK (all ABIs), which
takes much longer to compile. `arm64-v8a` covers virtually every current Android phone —
but the **emulator needs `x86_64`**:

```bash
./gradlew assembleRelease -PreactNativeArchitectures=x86_64
```

On the emulator, set the front camera to the computer webcam, otherwise the app opens but
sees no faces. In `~/.android/avd/<name>.avd/config.ini`:

```ini
hw.camera.front=webcam0
hw.camera.back=virtualscene
```

Setting both to `webcam0` does **not** work: only one camera is created, and CameraX
refuses to initialize when the other is missing (`CameraIdListIncorrectException`).

> This APK is signed with the **debug key**: it installs and runs on any device but is not
> suitable for the Play Store. To publish, generate your own upload key and configure
> `signingConfigs.release`, or use `eas build`.

## Cloud Builds and OTA Updates

```bash
# cloud build, no local SDK needed
npx eas build --platform android --profile preview

# OTA update (JavaScript only): logic/text fixes without rebuilding
npx eas update --channel production --message "change description"
```

The channels (`development` / `preview` / `production`) live in `eas.json`. OTA updates
**JavaScript only**; native changes, icons, or permissions require a new APK.

## Testing Without a Device

```bash
npm run verify            # runs the four below in sequence

npm run typecheck         # tsc --noEmit
npm run verify:math       # coordinate rotation and cosine similarity
npm run verify:alignment  # alignment warp vs. a reference implementation
npm run verify:worklets   # forbidden globals inside worklets
```

These tests exist because errors in this area do not break anything visibly: if
coordinate conversion or the warp are wrong, the crop grabs the wrong part of the frame
and recognition silently becomes noise.

- `verify:math` — round-trip of the 4 orientations against an independent rotation model,
  plus crop edge cases.
- `verify:alignment` — loads the actual `.ts` files from `src/face/` and checks they
  produce the **same 112x112 tensor**, value by value, as the Python reference. It
  depends on local fixtures generated by
  [scripts/offline-eval](scripts/offline-eval/README.md) (not versioned, since they
  derive from real photos); without them the test is skipped.
- `verify:worklets` — scans for JS global functions inside `'worklet'` blocks, which
  worklets-core cannot share (fails only on the device).

## Usage

1. Open the app and grant the camera permission.
2. **Enroll** in one of these ways:
   - tap **`+`** (top left) and pick a gallery photo or take one on the spot;
   - **or** point the camera at the person and tap the **capture circle** (bottom bar) —
     the scene photo becomes the enrollment.
   - Use a sharp, front-facing photo with **only the person** to enroll.
3. Fill in **name** and **date of birth** (`DD/MM/YYYY`) and save.
4. Return to the camera: the face now shows a **green** box and the name.
5. **Zoom:** pinch with two fingers or the **+ / −** buttons on the side; tap the level
   (e.g., `1.0x`) to reset.
6. **Database** (bottom bar): lists enrollments. Tap a person to **edit** name, date, or
   photo, or use **Remover** (Remove).

## Configuration

Everything is in [`src/face/constants.ts`](src/face/constants.ts):

| Constant | Default | Purpose |
| --- | --- | --- |
| `MATCH_THRESHOLD` | `0.62` | Minimum cosine to call it "the same person". Calibrated: in tests, same person scored >= 0.784 and different people <= 0.448 — any value between ~0.50 and ~0.75 classifies everything correctly. |
| `DETECTION_FPS` | `8` | Detection frequency. Lower = more battery and less heat. |
| `RECHECK_EVERY_TICKS` | `16` | How many cycles before re-confirming an already-identified face (16 is ~2 s). |
| `MIN_FACE_PIXELS` | `64` | Below this the face turns yellow instead of being recognized. |
| `MAX_YAW_ANGLE` / `MAX_PITCH_ANGLE` | `45` / `40` | How far the head may turn before it turns yellow. |

One photo per person already works well — in tests, the same person shot on different
days, lighting, and cameras scored about 0.86 similarity. For extra robustness, the
natural path is to store **several** photos per person and match against the best result.

> Stored embeddings are only comparable if they came from the same preprocessing. That is
> why the database key carries a version (`.../people/v2-aligned`): when the crop changes,
> old enrollments are discarded instead of silently degrading recognition.

## Project Structure

```
App.tsx                        global state (database + screens) and the enrollment model
src/
  db/people.ts                 AsyncStorage CRUD, age, and date formatting
  face/
    constants.ts               thresholds and tunable parameters
    embedding.ts               normalization, alignment warp, L2 and cosine (worklets)
    geometry.ts                frame <-> screen rotation and aligned region (worklets)
    enroll.ts                  photo enrollment: image -> face -> embedding
  screens/
    CameraScreen.tsx           camera, zoom, capture, frame processor, and recognition
    AddPersonScreen.tsx        enrollment/edit form
    PeopleScreen.tsx           database list (edit / remove)
  components/FaceOverlay.tsx   colored boxes and labels
assets/models/mobilefacenet.tflite
scripts/
  generate-icons.py            generates the app icons from code
  verify-*.js                  device-free tests (see above)
  offline-eval/                Python quality evaluation (see its own README)
```

## Privacy

Photos and face vectors stay **only** in the app's internal storage; nothing is sent to
any server. Biometric data is sensitive personal data: if you use this with other people,
obtain their consent. The test images used during development are **not** part of this
repository, precisely because they are real people's faces.

## Credits

- MobileFaceNet model (TFLite) from
  [estebanuri/face_recognition](https://github.com/estebanuri/face_recognition).
- Face detection: ML Kit via
  [react-native-vision-camera-face-detector](https://github.com/luicfrr/react-native-vision-camera-face-detector).
- Camera and frame processors:
  [react-native-vision-camera](https://github.com/mrousavy/react-native-vision-camera) and
  [react-native-fast-tflite](https://github.com/mrousavy/react-native-fast-tflite).

## License

Released under the [MIT License](LICENSE).
