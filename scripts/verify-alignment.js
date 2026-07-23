/**
 * Confere o alinhamento facial contra fixtures geradas por uma implementacao
 * de referencia independente (Python/OpenCV), rodada sobre fotos reais.
 *
 * O que isso pega: erro de porte na matematica do warp. As funcoes de
 * `src/face` rodam dentro de worklets no aparelho, onde nao da para inspecionar
 * nada; aqui elas rodam em Node contra numeros conferidos.
 *
 * As fixtures ficam em scripts/fixtures/ (geradas por scripts/offline-eval).
 * Rode com `npm run verify:alignment`.
 */
const path = require('path');
const fs = require('fs');
const babel = require('@babel/core');

const projectRoot = path.join(__dirname, '..');
const FIXTURES = path.join(__dirname, 'fixtures');

const moduleCache = new Map();

function loadTsModule(relativePath) {
  const filename = path.resolve(projectRoot, relativePath);
  if (moduleCache.has(filename)) return moduleCache.get(filename);

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

if (!fs.existsSync(path.join(FIXTURES, 'cases.json'))) {
  console.log('fixtures ausentes em scripts/fixtures/ — veja scripts/offline-eval/README.md');
  process.exit(0);
}

const { eyeAlignedRegion } = loadTsModule('src/face/geometry.ts');
const { warpRegionToTensor } = loadTsModule('src/face/embedding.ts');

const spec = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'cases.json'), 'utf8'));
const { regionSize, modelInputSize, cases } = spec;

let failures = 0;
const check = (label, ok, detail = '') => {
  if (ok) console.log(`ok     ${label}`);
  else {
    failures++;
    console.log(`FALHA  ${label}  ${detail}`);
  }
};

for (const c of cases) {
  // 1) A regiao calculada bate com a referencia?
  const region = eyeAlignedRegion(
    { x: c.rightEye[0], y: c.rightEye[1] },
    { x: c.leftEye[0], y: c.leftEye[1] },
    c.imageWidth,
    c.imageHeight
  );
  const regionOk =
    region != null &&
    region.x === c.region.x &&
    region.y === c.region.y &&
    region.width === c.region.width &&
    region.height === c.region.height;
  check(
    `${c.name}: regiao alinhada`,
    regionOk,
    `esperado ${JSON.stringify(c.region)} obtido ${JSON.stringify(region)}`
  );
  if (!regionOk) continue;

  // 2) O warp produz o mesmo tensor?
  const regionBuffer = new Uint8Array(
    fs.readFileSync(path.join(FIXTURES, `${c.name}.region.bin`))
  );
  const expectedBuffer = fs.readFileSync(path.join(FIXTURES, `${c.name}.tensor.bin`));
  const expected = new Float32Array(
    expectedBuffer.buffer,
    expectedBuffer.byteOffset,
    expectedBuffer.length / 4
  );

  const [rex, rey, lex, ley] = c.eyesInRegion;
  const actual = warpRegionToTensor(regionBuffer, regionSize, rex, rey, lex, ley);

  if (actual == null) {
    check(`${c.name}: warp`, false, 'devolveu null');
    continue;
  }

  const expectedLength = modelInputSize * modelInputSize * 3;
  check(`${c.name}: tamanho do tensor`, actual.length === expectedLength,
    `${actual.length} != ${expectedLength}`);

  let maxDiff = 0;
  let sumSq = 0;
  for (let i = 0; i < expectedLength; i++) {
    const d = Math.abs(actual[i] - expected[i]);
    if (d > maxDiff) maxDiff = d;
    sumSq += d * d;
  }
  const rms = Math.sqrt(sumSq / expectedLength);
  // Tolerancia frouxa em float32: a diferenca real fica varias ordens abaixo.
  check(
    `${c.name}: tensor identico ao de referencia (maxdiff=${maxDiff.toExponential(2)}, rms=${rms.toExponential(2)})`,
    maxDiff < 1e-4
  );
}

console.log(failures === 0 ? '\nTODOS OS TESTES PASSARAM' : `\n${failures} FALHA(S)`);
process.exit(failures === 0 ? 0 : 1);
