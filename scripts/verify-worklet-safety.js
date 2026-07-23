/**
 * Procura chamadas a funcoes globais do JavaScript dentro de codigo marcado
 * como `'worklet'`.
 *
 * Motivo: o runtime paralelo do react-native-worklets-core nao consegue
 * compartilhar funcoes JS comuns vindas do runtime principal. Chamar uma delas
 * dentro de um worklet lanca, em tempo de execucao e so no aparelho:
 *
 *   "Regular javascript function 'isFinite' cannot be shared.
 *    Try decorating the function with the 'worklet' keyword..."
 *
 * Isso custou caro uma vez: `isFinite` no meio do alinhamento facial fazia todo
 * rosto cair no `catch` e ficar amarelo. O typecheck passava, o bundle passava,
 * os testes offline passavam — porque na thread JS a funcao existe.
 *
 * Rode com `npm run verify:worklets`.
 */
const fs = require('fs');
const path = require('path');

const projectRoot = path.join(__dirname, '..');
const SRC = path.join(projectRoot, 'src');

/**
 * Globais que existem no runtime principal mas nao sao compartilhaveis.
 * `Math.*` e os construtores de TypedArray funcionam e ficam de fora.
 */
const FORBIDDEN = [
  'isFinite',
  'isNaN',
  'parseInt',
  'parseFloat',
  'decodeURI',
  'decodeURIComponent',
  'encodeURI',
  'encodeURIComponent',
];

/** Alternativas seguras, para a mensagem de erro ser acionavel. */
const SUGGESTIONS = {
  isFinite: 'use `x * 0 === 0`',
  isNaN: 'use `x !== x`',
  parseInt: 'use `Math.trunc(Number(x))` ou converta fora do worklet',
  parseFloat: 'use `Number(x)` ou converta fora do worklet',
};

function listFiles(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...listFiles(full));
    else if (/\.tsx?$/.test(entry.name)) out.push(full);
  }
  return out;
}

/**
 * Marca as linhas que pertencem a alguma funcao com a diretiva 'worklet'.
 * Aproximacao por contagem de chaves: basta para arquivos deste tamanho e nao
 * exige um parser completo.
 */
function workletLineRanges(source) {
  const lines = source.split('\n');
  const ranges = [];
  for (let i = 0; i < lines.length; i++) {
    if (!/['"]worklet['"];?\s*$/.test(lines[i].trim())) continue;

    // Sobe ate a abertura do bloco que contem a diretiva.
    let depth = 1;
    let end = lines.length - 1;
    for (let j = i + 1; j < lines.length; j++) {
      for (const ch of lines[j]) {
        if (ch === '{') depth++;
        else if (ch === '}') depth--;
      }
      if (depth <= 0) {
        end = j;
        break;
      }
    }
    ranges.push([i, end]);
  }
  return ranges;
}

let failures = 0;
let scanned = 0;
let workletBlocks = 0;

for (const file of listFiles(SRC)) {
  const source = fs.readFileSync(file, 'utf8');
  if (!source.includes('worklet')) continue;
  scanned++;

  const ranges = workletLineRanges(source);
  workletBlocks += ranges.length;
  const lines = source.split('\n');

  for (const [start, end] of ranges) {
    for (let i = start; i <= end; i++) {
      const line = lines[i];
      if (line.trim().startsWith('//') || line.trim().startsWith('*')) continue;

      for (const name of FORBIDDEN) {
        // chamada da funcao, e nao `Foo.isFinite` nem `x.isNaN`
        const called = new RegExp(`(^|[^.\\w])${name}\\s*\\(`).test(line);
        if (!called) continue;

        failures++;
        const rel = path.relative(projectRoot, file).replace(/\\/g, '/');
        const hint = SUGGESTIONS[name] ? ` — ${SUGGESTIONS[name]}` : '';
        console.log(`FALHA  ${rel}:${i + 1}  chama '${name}' dentro de worklet${hint}`);
        console.log(`         ${line.trim()}`);
      }
    }
  }
}

console.log(
  failures === 0
    ? `\nok — ${workletBlocks} blocos worklet em ${scanned} arquivos, nenhum global proibido`
    : `\n${failures} problema(s) encontrado(s)`
);
process.exit(failures === 0 ? 0 : 1);
