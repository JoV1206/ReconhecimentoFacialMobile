/**
 * Confere a matematica de rotacao e de similaridade sem precisar de um aparelho.
 *
 * A conversao de coordenadas e a parte mais facil de errar de forma silenciosa:
 * um erro ali nao quebra nada, so faz o recorte pegar o lugar errado do frame e
 * o reconhecimento passar a devolver ruido. Rodar com `npm run verify:math`.
 *
 * Modelo de referencia: girar o buffer BRUTO (W x H) no sentido horario por R
 * graus produz a imagem "em pe". O ML Kit devolve as caixas nesse espaco em pe,
 * entao `uprightRectToFrameRect` precisa ser exatamente a inversa disso.
 */
const path = require('path');
const fs = require('fs');
const babel = require('@babel/core');

const projectRoot = path.join(__dirname, '..');

// Mini carregador de TypeScript: transpila e resolve os imports relativos entre
// os proprios arquivos de src/face (embedding.ts importa ./constants).
const moduleCache = new Map();

function loadTsModule(relativePath) {
  const filename = path.resolve(projectRoot, relativePath);
  const cached = moduleCache.get(filename);
  if (cached) return cached;

  const source = fs.existsSync(filename) ? filename : `${filename}.ts`;
  const { code } = babel.transformSync(fs.readFileSync(source, 'utf8'), {
    filename: source,
    presets: [[require.resolve('@babel/preset-typescript'), { onlyRemoveTypeImports: false }]],
    plugins: [require.resolve('@babel/plugin-transform-modules-commonjs')],
    babelrc: false,
    configFile: false,
  });

  const loaded = { exports: {} };
  moduleCache.set(filename, loaded.exports);

  const localRequire = (request) =>
    request.startsWith('.')
      ? loadTsModule(path.resolve(path.dirname(source), request))
      : require(request);

  new Function('module', 'exports', 'require', code)(loaded, loaded.exports, localRequire);
  moduleCache.set(filename, loaded.exports);
  return loaded.exports;
}

const { uprightRectToFrameRect, uprightFrameSize, orientationToUprightRotation, toSquareFaceBox } =
  loadTsModule('src/face/geometry.ts');
const { l2Normalize, cosineSimilarity } = loadTsModule('src/face/embedding.ts');

let failures = 0;

function check(label, condition, detail = '') {
  if (condition) {
    console.log(`ok     ${label}`);
  } else {
    failures++;
    console.log(`FALHA  ${label}  ${detail}`);
  }
}

/** Mapeia um ponto do buffer bruto para o espaco em pe, girando R no horario. */
function rawPointToUpright(x, y, R, W, H) {
  switch (R) {
    case 90:
      return { x: H - 1 - y, y: x };
    case 180:
      return { x: W - 1 - x, y: H - 1 - y };
    case 270:
      return { x: y, y: W - 1 - x };
    default:
      return { x, y };
  }
}

// --- 1. frame.orientation -> graus -------------------------------------------
// Espelha Orientation.reversed() do Android: rotationDegrees do CameraX
// equivale a (360 - graus da orientation) % 360.
check('portrait -> 0', orientationToUprightRotation('portrait') === 0);
check('landscape-right -> 90', orientationToUprightRotation('landscape-right') === 90);
check('portrait-upside-down -> 180', orientationToUprightRotation('portrait-upside-down') === 180);
check('landscape-left -> 270', orientationToUprightRotation('landscape-left') === 270);

// --- 2. Ida e volta bruto -> em pe -> bruto -----------------------------------
const W = 1280;
const H = 720;

for (const R of [0, 90, 180, 270]) {
  const upright = uprightFrameSize(W, H, R);
  const raw = { x: 200, y: 100, width: 160, height: 160 };

  const corners = [
    rawPointToUpright(raw.x, raw.y, R, W, H),
    rawPointToUpright(raw.x + raw.width - 1, raw.y, R, W, H),
    rawPointToUpright(raw.x, raw.y + raw.height - 1, R, W, H),
    rawPointToUpright(raw.x + raw.width - 1, raw.y + raw.height - 1, R, W, H),
  ];
  const xs = corners.map((c) => c.x);
  const ys = corners.map((c) => c.y);
  const uprightRect = {
    x: Math.min(...xs),
    y: Math.min(...ys),
    width: Math.max(...xs) - Math.min(...xs) + 1,
    height: Math.max(...ys) - Math.min(...ys) + 1,
  };

  check(
    `R=${R} caixa em pe cabe no frame em pe`,
    uprightRect.x >= 0 &&
      uprightRect.y >= 0 &&
      uprightRect.x + uprightRect.width <= upright.width &&
      uprightRect.y + uprightRect.height <= upright.height,
    JSON.stringify({ uprightRect, upright })
  );

  const back = uprightRectToFrameRect(uprightRect, R, W, H);
  check(
    `R=${R} ida e volta reconstroi o retangulo bruto`,
    back.x === raw.x && back.y === raw.y && back.width === raw.width && back.height === raw.height,
    `esperado ${JSON.stringify(raw)} obtido ${JSON.stringify(back)}`
  );

  check(
    `R=${R} recorte cabe no buffer bruto`,
    back.x >= 0 && back.y >= 0 && back.x + back.width <= W && back.y + back.height <= H,
    JSON.stringify(back)
  );
}

// --- 3. Recorte quadrado ------------------------------------------------------
// Rosto no meio do frame: vira quadrado com margem, com origem e lado pares.
const centered = toSquareFaceBox({ x: 300, y: 500, width: 100, height: 140 }, 1.25, 720, 1280);
check('recorte central e quadrado', centered.width === centered.height, JSON.stringify(centered));
check(
  'origem e lado sao pares (exigencia do YUV 4:2:0)',
  centered.x % 2 === 0 && centered.y % 2 === 0 && centered.width % 2 === 0,
  JSON.stringify(centered)
);
check(
  'recorte central cabe no frame',
  centered.x >= 0 &&
    centered.y >= 0 &&
    centered.x + centered.width <= 720 &&
    centered.y + centered.height <= 1280,
  JSON.stringify(centered)
);

// Rosto colado na borda: precisa encolher em vez de estourar o limite.
const edge = toSquareFaceBox({ x: 0, y: 0, width: 120, height: 120 }, 1.25, 720, 1280);
check(
  'rosto na borda nao estoura o frame',
  edge !== null && edge.x >= 0 && edge.y >= 0 && edge.x + edge.width <= 720,
  JSON.stringify(edge)
);

// Centro fora do frame nao tem recorte valido.
check('centro fora do frame devolve null', toSquareFaceBox({ x: -400, y: 10, width: 100, height: 100 }, 1.25, 720, 1280) === null);

// --- 4. Similaridade ----------------------------------------------------------
const a = l2Normalize([3, 4]);
check('l2Normalize devolve vetor unitario', Math.abs(Math.hypot(a[0], a[1]) - 1) < 1e-9, JSON.stringify(a));
check('vetores iguais -> cosseno 1', Math.abs(cosineSimilarity(a, a) - 1) < 1e-9);
check(
  'vetores ortogonais -> cosseno 0',
  Math.abs(cosineSimilarity(l2Normalize([1, 0]), l2Normalize([0, 1]))) < 1e-9
);
check(
  'vetores opostos -> cosseno -1',
  Math.abs(cosineSimilarity(l2Normalize([1, 0]), l2Normalize([-1, 0])) + 1) < 1e-9
);
check('vetor zero nao vira NaN', l2Normalize([0, 0]).every((v) => v === 0));

console.log(failures === 0 ? '\nTODOS OS TESTES PASSARAM' : `\n${failures} FALHA(S)`);
process.exit(failures === 0 ? 0 : 1);
