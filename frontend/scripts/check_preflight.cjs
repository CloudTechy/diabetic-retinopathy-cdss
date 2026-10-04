/**
 * Execute the browser pre-check (src/utils/retinalValidator.ts) on fixed inputs
 * and assert every field it returns.
 *
 * Why this exists: the pre-check once reported "Aperture coverage ... within
 * thresholds" without computing a coverage, reported Gate 3 as passed without
 * evaluating it, and returned a "Signature Valid" metric, a focus estimate of
 * 180.0 and colour figures of 1.00 / 33% that it had never computed. Rules that
 * read the source could not see any of that; running the code does.
 *
 * No test runner and no new dependency: the validator is bundled with the
 * esbuild that vite already installs, and run in Node against a stub canvas
 * fed with synthetic 256x256 pixel data built below. The inputs are synthetic
 * patterns, not photographs, and are labelled as such.
 *
 * Usage (after `npm ci`):   node scripts/check_preflight.cjs
 * Exit status 0 only if every case behaves as asserted.
 */
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const esbuild = require('esbuild');

const SRC = path.resolve(__dirname, '..', 'src', 'utils');
const VALIDATOR = path.join(SRC, 'retinalValidator.ts');
const THRESHOLDS = path.join(SRC, 'validationThresholds.ts');

const norm = (p) => fs.readFileSync(p, 'utf8').replace(/\r\n/g, '\n');
const sourceHash = crypto.createHash('sha256')
  .update(norm(VALIDATOR) + '\n--\n' + norm(THRESHOLDS), 'utf8').digest('hex');

// ---- bundle the real validator
const bundled = esbuild.buildSync({
  entryPoints: [VALIDATOR], bundle: true, format: 'cjs', platform: 'node', write: false, logLevel: 'silent',
}).outputFiles[0].text;

// ---- synthetic inputs (256x256 RGBA)
const N = 256;
function pixels(fn) {
  const d = new Uint8ClampedArray(N * N * 4);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const [r, g, b] = fn(x, y);
    const i = (y * N + x) * 4;
    d[i] = r; d[i + 1] = g; d[i + 2] = b; d[i + 3] = 255;
  }
  return d;
}
const inDisc = (x, y, radius) => (x - 128) ** 2 + (y - 128) ** 2 <= radius * radius;
// a reddish disc on a dark frame, with a little texture so a gradient exists
const DISC_REDDISH = pixels((x, y) => inDisc(x, y, 120) ? [170 + ((x * 7 + y * 3) % 40), 80 + ((x + y) % 20), 40] : [0, 0, 0]);
// the same colours in a disc covering only ~8% of the frame: coverage is the ONLY criterion it fails
const SMALL_DISC_REDDISH = pixels((x, y) => inDisc(x, y, 40) ? [170 + ((x * 7 + y * 3) % 40), 80 + ((x + y) % 20), 40] : [0, 0, 0]);
const DISC_GREY = pixels((x, y) => inDisc(x, y, 120) ? [120, 120, 120] : [0, 0, 0]);
const ALL_DARK = pixels(() => [0, 0, 0]);
// light page with dark "text" rows; the rows avoid y = 0 and y = 255 so all four corners are bright
const BRIGHT_NEUTRAL = pixels((x, y) => (y % 16 >= 6 && y % 16 < 8 ? [60, 60, 60] : [235, 235, 235]));

// ---- run the validator against a stub canvas
function run(data, { width = 1024, height = 1024, mime = 'image/png', size = 900000, canvas = true } = {}) {
  const sandboxDocument = {
    createElement: () => ({
      width: 0, height: 0,
      getContext: () => (canvas ? { drawImage() {}, getImageData: () => ({ data }) } : null),
    }),
  };
  const mod = { exports: {} };
  new Function('module', 'exports', 'document', bundled)(mod, mod.exports, sandboxDocument);
  const img = { naturalWidth: width, naturalHeight: height, width, height };
  return mod.exports.analyzeRetinalImageOnCanvas(img, size, mime);
}

const failures = [];
function check(name, result, expectations) {
  const problems = [];
  for (const [label, ok] of expectations(result)) if (!ok) problems.push(label);
  // invariants of EVERY result
  if (result.gate3.status !== 'notEvaluated') problems.push('gate3.status is not notEvaluated');
  if ('passed' in result.gate3) problems.push('gate3 carries a passed field');
  if (result.preflightPassed !== (result.gate1.passed && result.gate2.passed)) problems.push('preflightPassed is not gate1 && gate2');
  if (/signature valid/i.test(result.gate1.metric)) problems.push('gate1 claims a signature check the browser does not perform');
  if (!result.gate2.passed && result.gate3.focusEstimate !== null) problems.push('a focus estimate is reported although none was computed');
  if (/within thresholds/.test(result.gate2.metric) && !result.gate2.passed) problems.push('"within thresholds" on a failed Gate 2');
  if (!result.gate2.coverageEvaluated && (result.gate2.redToBlueRatio !== null || result.gate2.redShare !== null)) {
    problems.push('colour figures reported without a canvas');
  }
  const cov = result.gate2.coverageEvaluated ? (result.gate2.foregroundCoverage * 100).toFixed(1) + '%' : 'not evaluated';
  const summary = `preflight=${result.preflightPassed} failedGate=${result.failedGate} coverage=${cov} gate3=${result.gate3.status}`;
  if (problems.length) { failures.push(name); console.log(`  FAIL ${name}: ${summary}\n        ` + problems.join('\n        ')); }
  else console.log(`  ok   ${name}: ${summary}`);
}

console.log(`PREFLIGHT CHECK source sha256 ${sourceHash}`);
console.log('  (retinalValidator.ts + validationThresholds.ts, executed in Node against synthetic 256x256 patterns)');

check('reddish disc on a dark frame (synthetic)', run(DISC_REDDISH), (r) => [
  ['passes gates 1 and 2', r.preflightPassed === true && r.failedGate === null],
  ['coverage is the measured disc share (~69%)', Math.abs(r.gate2.foregroundCoverage - Math.PI * 120 * 120 / (N * N)) < 0.01],
  ['coverage evaluated', r.gate2.coverageEvaluated === true],
  ['colour figures computed', r.gate2.redToBlueRatio > 3 && r.gate2.redShare > 0.5],
  ['metric says within thresholds', /within thresholds/.test(r.gate2.metric)],
  ['a focus estimate was computed', typeof r.gate3.focusEstimate === 'number'],
]);
check('all-dark frame', run(ALL_DARK), (r) => [
  ['fails Gate 2', r.preflightPassed === false && r.failedGate === 2],
  ['coverage 0% and evaluated', r.gate2.coverageEvaluated === true && r.gate2.foregroundCoverage === 0],
  ['reason names coverage', /Foreground coverage 0\.0% is below the configured minimum/.test(r.gate2.rejectionReason || '')],
  ['colour figures are the whole-frame values, as the server computes them', r.gate2.redToBlueRatio === 0 && r.gate2.redShare === 0],
]);
check('small reddish disc (coverage ~8%, colour profile valid)', run(SMALL_DISC_REDDISH), (r) => [
  ['fails Gate 2', r.preflightPassed === false && r.failedGate === 2],
  ['coverage is the measured disc share (~7.7%)', Math.abs(r.gate2.foregroundCoverage - Math.PI * 40 * 40 / (N * N)) < 0.005],
  ['the colour profile alone would pass, so coverage is the sole cause', r.gate2.redToBlueRatio > 3 && r.gate2.redShare > 0.5 && r.gate2.isDocumentOrDiagram === false],
  ['reason names coverage', /Foreground coverage 7\.\d% is below the configured minimum/.test(r.gate2.rejectionReason || '')],
  ['no "within thresholds"', !/within thresholds/.test(r.gate2.metric)],
]);
check('bright neutral frame (document-like)', run(BRIGHT_NEUTRAL), (r) => [
  ['fails Gate 2', r.preflightPassed === false && r.failedGate === 2],
  ['flagged as document/diagram', r.gate2.isDocumentOrDiagram === true],
  ['reason is the document reason', /document or diagram/.test(r.gate2.rejectionReason || '')],
]);
check('300x200 image', run(DISC_REDDISH, { width: 300, height: 200 }), (r) => [
  ['fails Gate 1', r.preflightPassed === false && r.failedGate === 1],
  ['reason names the minimum dimension', /minimum of \d+x\d+ px/.test(r.gate1.rejectionReason || '')],
]);
check('2400x800 panorama', run(DISC_REDDISH, { width: 2400, height: 800 }), (r) => [
  ['fails Gate 2 on aspect ratio', r.failedGate === 2 && /Aspect ratio 3\.00/.test(r.gate2.rejectionReason || '')],
]);
check('grey disc on a dark frame', run(DISC_GREY), (r) => [
  ['fails Gate 2 on colour profile', r.failedGate === 2 && /Colour profile outside/.test(r.gate2.rejectionReason || '')],
  ['coverage itself passes (~69%)', r.gate2.foregroundCoverage > 0.6],
]);
check('no canvas available', run(DISC_REDDISH, { canvas: false }), (r) => [
  ['cannot pass', r.preflightPassed === false && r.failedGate === 2],
  ['coverage not evaluated', r.gate2.coverageEvaluated === false],
  ['metric says not evaluated', /Not evaluated in the browser/.test(r.gate2.metric)],
  ['no colour figure and no focus estimate', r.gate2.redToBlueRatio === null && r.gate2.redShare === null && r.gate3.focusEstimate === null],
]);

const total = 8;
console.log(`PREFLIGHT CHECK: ${total - failures.length}/${total} cases as expected`);
process.exit(failures.length ? 1 : 0);
