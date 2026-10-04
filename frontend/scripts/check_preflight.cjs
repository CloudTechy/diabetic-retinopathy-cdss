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
 * Each criterion has an input that fails on that criterion ALONE, so removing
 * or weakening one rule turns exactly that case red. An earlier version of this
 * script had no such inputs for most rules and printed a hard-coded "8/8": a
 * reviewer removed six rules one at a time and it stayed green. The count
 * printed at the end is now the number of cases that actually ran.
 *
 * No test runner and no new dependency: the validator is bundled with the
 * esbuild that vite already installs, and run in Node against a stub canvas
 * fed with 256x256 pixel data. All inputs but one are synthetic patterns built
 * below, not photographs, and are labelled as such. The one photograph is the
 * held-out test fixture, exported to a 256x256 RGBA dump by
 * backend/scripts/export_preflight_fixture.py (a nearest-neighbour sample, not
 * the browser's own scaling); the figures expected for it are computed by that
 * Python script, not by this one.
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

const FIXTURE_SIDECAR = path.join(__dirname, 'fixtures', 'aptos_heldout_d1f1ea894da1_256.json');

const norm = (p) => fs.readFileSync(p, 'utf8').replace(/\r\n/g, '\n');
// The hash covers this script and the fixture's sidecar as well as the validator:
// a recorded run cannot be passed off as the run of a harness whose expectations
// were changed afterwards.
const sourceHash = crypto.createHash('sha256')
  .update([VALIDATOR, THRESHOLDS, __filename, FIXTURE_SIDECAR].map(norm).join('\n--\n'), 'utf8').digest('hex');

// the thresholds the validator is supposed to apply, read from the generated file
const T = {};
for (const m of norm(THRESHOLDS).matchAll(/export const (\w+) = ([\d.]+);/g)) T[m[1]] = Number(m[2]);

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
const dist2 = (x, y) => (x - 128) ** 2 + (y - 128) ** 2;
const inDisc = (x, y, radius) => dist2(x, y) <= radius * radius;
const share = (radius) => { let c = 0; for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (inDisc(x, y, radius)) c++; return c / (N * N); };
// reddish, with a little texture so a gradient exists; luminance stays above 100
const reddish = (x, y) => [170 + ((x * 7 + y * 3) % 40), 80 + ((x + y) % 20), 40];

const DISC_REDDISH = pixels((x, y) => inDisc(x, y, 120) ? reddish(x, y) : [0, 0, 0]);
// the same colours in a disc covering only ~8% of the frame: fails on coverage alone
const SMALL_DISC_REDDISH = pixels((x, y) => inDisc(x, y, 40) ? reddish(x, y) : [0, 0, 0]);
// a bright disc with a DIM ring (luminance ~40) around it: the ring counts as
// foreground only if the cut-off is the server's 15
const SOFT_EDGED_DISC = pixels((x, y) => inDisc(x, y, 120) ? reddish(x, y) : inDisc(x, y, 140) ? [70, 30, 15] : [0, 0, 0]);
// the reddish disc with a ring of ONE colour around it; whether the ring counts as
// foreground pins the cut-off and the luminance weights
const ringed = (rgb) => pixels((x, y) => inDisc(x, y, 120) ? reddish(x, y) : inDisc(x, y, 140) ? rgb : [0, 0, 0]);
// a uniform frame: every pixel, the four corners included, has this colour
const uniform = (rgb) => pixels(() => rgb);
const DISC_GREY = pixels((x, y) => inDisc(x, y, 120) ? [120, 120, 120] : [0, 0, 0]);
// R/B = 1.25 (above the minimum) but red share = 33.3% (below it): fails on red share alone
const DISC_LOW_RED_SHARE = pixels((x, y) => inDisc(x, y, 120) ? [100, 120, 80] : [0, 0, 0]);
// red share 37.5% (above the minimum) but R/B = 1.09 (below it): fails on R/B alone
const DISC_LOW_RB = pixels((x, y) => inDisc(x, y, 120) ? [120, 90, 110] : [0, 0, 0]);
// a full bright frame with a valid colour profile: bright corners, but not a document
const BRIGHT_REDDISH_FRAME = pixels((x, y) => [245, 170 + ((x + y) % 6), 110]);   // corner mean (r+g+b)/3 = 175, above the heuristic's 160
const ALL_DARK = pixels(() => [0, 0, 0]);
// light page with dark "text" rows; the rows avoid y = 0 and y = 255 so all four corners are bright
const BRIGHT_NEUTRAL = pixels((x, y) => (y % 16 >= 6 && y % 16 < 8 ? [60, 60, 60] : [235, 235, 235]));
// exactly K foreground pixels, from the centre outwards in raster order (corners stay dark)
function exactlyForeground(k) {
  const order = [];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) order.push([dist2(x, y), x, y]);
  order.sort((a, b) => a[0] - b[0] || a[2] - b[2] || a[1] - b[1]);
  const lit = new Set(order.slice(0, k).map(([, x, y]) => y * N + x));
  return pixels((x, y) => (lit.has(y * N + x) ? reddish(x, y) : [0, 0, 0]));
}
const MIN_COVERAGE_PIXELS = Math.ceil(T.RETINAL_MIN_COVERAGE * N * N);   // the fewest pixels that reach the minimum

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

let ran = 0;
const failures = [];
function check(name, result, expectations) {
  ran += 1;
  const problems = [];
  for (const [label, ok] of expectations(result)) if (!ok) problems.push(label);
  // invariants of EVERY result
  if (result.gate3.status !== 'notEvaluated') problems.push('gate3.status is not notEvaluated');
  if ('passed' in result.gate3) problems.push('gate3 carries a passed field');
  if (result.preflightPassed !== (result.gate1.passed && result.gate2.passed)) problems.push('preflightPassed is not gate1 && gate2');
  if (!result.gate1.passed && result.gate2.passed) problems.push('gate2.passed is true although Gate 1 failed (the server never reaches Gate 2)');
  if (/signature valid/i.test(result.gate1.metric)) problems.push('gate1 claims a signature check the browser does not perform');
  if (!result.gate2.passed && result.gate3.focusEstimate !== null) problems.push('a focus estimate is reported although none was computed');
  if (/within thresholds/.test(result.gate2.metric) && !result.gate2.passed) problems.push('"within thresholds" on a failed Gate 2');
  if (!result.gate2.coverageEvaluated && (result.gate2.redToBlueRatio !== null || result.gate2.redShare !== null)) {
    problems.push('colour figures reported without a canvas');
  }
  const cov = result.gate2.coverageEvaluated ? (result.gate2.foregroundCoverage * 100).toFixed(3) + '%' : 'not evaluated';
  const summary = `preflight=${result.preflightPassed} failedGate=${result.failedGate} coverage=${cov} gate3=${result.gate3.status}`;
  if (problems.length) { failures.push(name); console.log(`  FAIL ${name}: ${summary}\n        ` + problems.join('\n        ')); }
  else console.log(`  ok   ${name}: ${summary}`);
}
const near = (a, b, tol) => Math.abs(a - b) <= tol;
const failsGate = (r, gate) => r.preflightPassed === false && r.failedGate === gate;

console.log(`PREFLIGHT CHECK source sha256 ${sourceHash}`);
console.log('  (retinalValidator.ts + validationThresholds.ts + this script + the fixture sidecar; executed in Node against 256x256 inputs)');

// ---- the passing case
check('reddish disc on a dark frame', run(DISC_REDDISH), (r) => [
  ['passes gates 1 and 2', r.preflightPassed === true && r.failedGate === null],
  ['coverage is the measured disc share', near(r.gate2.foregroundCoverage, share(120), 1e-9)],
  ['coverage evaluated', r.gate2.coverageEvaluated === true],
  ['colour figures computed', r.gate2.redToBlueRatio > 3 && r.gate2.redShare > 0.5],
  ['metric says within thresholds', /within thresholds/.test(r.gate2.metric)],
  ['a focus estimate was computed', typeof r.gate3.focusEstimate === 'number'],
  ['gate 1 metric names the declared type', /^Declared type, size and dimensions within limits/.test(r.gate1.metric)],
]);

// ---- Gate 1: one input per rule
check('declared type image/gif', run(DISC_REDDISH, { mime: 'image/gif' }), (r) => [
  ['fails Gate 1 on the declared type', failsGate(r, 1) && /Unsupported MIME format/.test(r.gate1.rejectionReason || '')],
]);
check(`file of ${T.MAX_UPLOAD_SIZE_MB} MiB + 1 byte`, run(DISC_REDDISH, { size: T.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1 }), (r) => [
  ['fails Gate 1 on size', failsGate(r, 1) && /exceeds the \d+ MB limit/.test(r.gate1.rejectionReason || '')],
]);
check(`file of exactly ${T.MAX_UPLOAD_SIZE_MB} MiB`, run(DISC_REDDISH, { size: T.MAX_UPLOAD_SIZE_MB * 1024 * 1024 }), (r) => [
  ['passes (the limit is inclusive, as on the server)', r.preflightPassed === true],
]);
check('narrow image: width below the minimum, height above', run(DISC_REDDISH, { width: T.MIN_IMAGE_DIMENSION - 1, height: 600 }), (r) => [
  ['fails Gate 1 on width alone', failsGate(r, 1) && /minimum of \d+x\d+ px/.test(r.gate1.rejectionReason || '')],
]);
check('short image: height below the minimum, width above', run(DISC_REDDISH, { width: 600, height: T.MIN_IMAGE_DIMENSION - 1 }), (r) => [
  ['fails Gate 1 on height alone', failsGate(r, 1) && /minimum of \d+x\d+ px/.test(r.gate1.rejectionReason || '')],
]);
check('image of exactly the minimum dimension', run(DISC_REDDISH, { width: T.MIN_IMAGE_DIMENSION, height: T.MIN_IMAGE_DIMENSION }), (r) => [
  ['passes', r.preflightPassed === true],
]);
check('undeclared MIME type', run(DISC_REDDISH, { mime: '' }), (r) => [
  ['reports the type as not declared', r.gate1.mime === 'not declared'],
  ['gate 1 metric does not claim a declared type', /^Type not declared/.test(r.gate1.metric)],
]);

// ---- Gate 2: aspect ratio, both bounds
check('2400x800 panorama (aspect 3.00)', run(DISC_REDDISH, { width: 2400, height: 800 }), (r) => [
  ['fails Gate 2 on the upper aspect bound', failsGate(r, 2) && /Aspect ratio 3\.00/.test(r.gate2.rejectionReason || '')],
]);
check('600x1000 portrait (aspect 0.60)', run(DISC_REDDISH, { width: 600, height: 1000 }), (r) => [
  ['fails Gate 2 on the lower aspect bound', failsGate(r, 2) && /Aspect ratio 0\.60/.test(r.gate2.rejectionReason || '')],
]);
// the bounds themselves, and one pixel beyond each
check('1650x1000 (aspect exactly 1.65)', run(DISC_REDDISH, { width: 1650, height: 1000 }), (r) => [
  ['passes: the upper bound is inclusive', r.preflightPassed === true],
]);
check('1651x1000 (aspect just above 1.65)', run(DISC_REDDISH, { width: 1651, height: 1000 }), (r) => [
  ['fails Gate 2 on the upper aspect bound', failsGate(r, 2) && /Aspect ratio/.test(r.gate2.rejectionReason || '')],
]);
check('650x1000 (aspect exactly 0.65)', run(DISC_REDDISH, { width: 650, height: 1000 }), (r) => [
  ['passes: the lower bound is inclusive', r.preflightPassed === true],
]);
check('649x1000 (aspect just below 0.65)', run(DISC_REDDISH, { width: 649, height: 1000 }), (r) => [
  ['fails Gate 2 on the lower aspect bound', failsGate(r, 2) && /Aspect ratio/.test(r.gate2.rejectionReason || '')],
]);

// ---- Gate 2: coverage
check('small reddish disc (colour profile valid)', run(SMALL_DISC_REDDISH), (r) => [
  ['fails Gate 2', failsGate(r, 2)],
  ['coverage is the measured disc share', near(r.gate2.foregroundCoverage, share(40), 1e-9)],
  ['the colour profile alone would pass, so coverage is the sole cause', r.gate2.redToBlueRatio > 3 && r.gate2.redShare > 0.5 && r.gate2.isDocumentOrDiagram === false],
  ['reason names coverage', /Foreground coverage \d+\.\d% is below the configured minimum/.test(r.gate2.rejectionReason || '')],
]);
check('coverage one pixel above the minimum', run(exactlyForeground(MIN_COVERAGE_PIXELS)), (r) => [
  ['passes', r.preflightPassed === true],
  ['coverage is exactly the lit share', near(r.gate2.foregroundCoverage, MIN_COVERAGE_PIXELS / (N * N), 1e-12)],
]);
check('coverage one pixel below the minimum', run(exactlyForeground(MIN_COVERAGE_PIXELS - 1)), (r) => [
  ['fails Gate 2 on coverage alone', failsGate(r, 2) && /Foreground coverage/.test(r.gate2.rejectionReason || '')],
  ['the colour profile alone would pass', r.gate2.redToBlueRatio > 3 && r.gate2.redShare > 0.5],
]);
check('bright disc with a dim ring (luminance ~40)', run(SOFT_EDGED_DISC), (r) => [
  ['the dim ring counts as foreground (cut-off 15, as on the server)', near(r.gate2.foregroundCoverage, share(140), 1e-9)],
  ['and not only the bright disc', r.gate2.foregroundCoverage > share(120) + 0.1],
]);
// saturated pixels are foreground once, like any other
check('reddish disc with a saturated highlight (255,255,255)', run(pixels((x, y) => inDisc(x, y, 30) ? [255, 255, 255] : inDisc(x, y, 120) ? reddish(x, y) : [0, 0, 0])), (r) => [
  ['coverage is exactly the disc share', near(r.gate2.foregroundCoverage, share(120), 1e-12)],
]);
// the cut-off itself: luminance exactly 15 is background, 16 is foreground
check('ring of luminance exactly 15 (grey 15)', run(ringed([15, 15, 15])), (r) => [
  ['the ring is background: the cut-off is exclusive', near(r.gate2.foregroundCoverage, share(120), 1e-9)],
]);
check('ring of luminance 16 (grey 16)', run(ringed([16, 16, 16])), (r) => [
  ['the ring is foreground', near(r.gate2.foregroundCoverage, share(140), 1e-9)],
]);
// the weights: green counts for more than blue, as on the server
check('green ring (0,30,0): weighted luminance 17.6, plain mean 10', run(ringed([0, 30, 0])), (r) => [
  ['the ring is foreground', near(r.gate2.foregroundCoverage, share(140), 1e-9)],
]);
check('blue ring (0,0,60): weighted luminance 6.8, plain mean 20', run(ringed([0, 0, 60])), (r) => [
  ['the ring is background', near(r.gate2.foregroundCoverage, share(120), 1e-9)],
]);
check('all-dark frame', run(ALL_DARK), (r) => [
  ['fails Gate 2', failsGate(r, 2)],
  ['coverage 0% and evaluated', r.gate2.coverageEvaluated === true && r.gate2.foregroundCoverage === 0],
  ['reason names coverage', /Foreground coverage 0\.0% is below the configured minimum/.test(r.gate2.rejectionReason || '')],
  ['colour figures are the whole-frame values, as the server computes them', r.gate2.redToBlueRatio === 0 && r.gate2.redShare === 0],
]);

// ---- Gate 2: colour profile, each half alone
check('grey disc (R/B 1.00)', run(DISC_GREY), (r) => [
  ['fails Gate 2 on colour profile', failsGate(r, 2) && /Colour profile outside/.test(r.gate2.rejectionReason || '')],
  ['R/B is below the minimum', r.gate2.redToBlueRatio < T.RETINAL_RED_RATIO_MIN],
  ['coverage itself passes', r.gate2.foregroundCoverage > T.RETINAL_MIN_COVERAGE],
]);
check('disc with R/B above the minimum but red share below it', run(DISC_LOW_RED_SHARE), (r) => [
  ['R/B alone would pass', r.gate2.redToBlueRatio >= T.RETINAL_RED_RATIO_MIN],
  ['red share is below the minimum', r.gate2.redShare < T.RETINAL_RED_SHARE_MIN],
  ['fails Gate 2 on colour profile (red share alone)', failsGate(r, 2) && /Colour profile outside/.test(r.gate2.rejectionReason || '')],
]);

check('disc with red share above the minimum but R/B below it', run(DISC_LOW_RB), (r) => [
  ['red share alone would pass', r.gate2.redShare >= T.RETINAL_RED_SHARE_MIN],
  ['R/B is below the minimum', r.gate2.redToBlueRatio < T.RETINAL_RED_RATIO_MIN],
  ['not flagged as a document (corners are dark)', r.gate2.isDocumentOrDiagram === false],
  ['fails Gate 2 on colour profile (R/B alone)', failsGate(r, 2) && /Colour profile outside/.test(r.gate2.rejectionReason || '')],
]);

// ---- Gate 2: the browser-only document heuristic
check('bright neutral frame (document-like)', run(BRIGHT_NEUTRAL), (r) => [
  ['fails Gate 2', failsGate(r, 2)],
  ['flagged as document/diagram', r.gate2.isDocumentOrDiagram === true],
  ['reason is the document reason', /document or diagram/.test(r.gate2.rejectionReason || '')],
]);

check('bright reddish frame (bright corners, valid colour profile)', run(BRIGHT_REDDISH_FRAME), (r) => [
  ['not flagged as a document: the heuristic needs a neutral colour profile too', r.gate2.isDocumentOrDiagram === false],
  ['passes gates 1 and 2', r.preflightPassed === true],
  ['coverage 100%', r.gate2.foregroundCoverage === 1],
]);
// the corner threshold itself: a neutral frame at 160 is not a document, at 161 it is
check('uniform neutral frame at 160', run(uniform([160, 160, 160])), (r) => [
  ['not flagged as a document: the corner threshold is exclusive', r.gate2.isDocumentOrDiagram === false],
  ['fails Gate 2 on colour profile instead', failsGate(r, 2) && /Colour profile outside/.test(r.gate2.rejectionReason || '')],
]);
check('uniform neutral frame at 161', run(uniform([161, 161, 161])), (r) => [
  ['flagged as document/diagram', r.gate2.isDocumentOrDiagram === true && /document or diagram/.test(r.gate2.rejectionReason || '')],
]);

// ---- the held-out fixture (a photograph; nearest-neighbour 256x256 sample)
const SIDE = JSON.parse(norm(FIXTURE_SIDECAR));
const DUMP = fs.readFileSync(path.resolve(__dirname, '..', '..', SIDE.dump));
const DUMP_HASH = crypto.createHash('sha256').update(DUMP).digest('hex');
const E = SIDE.expected;
const P = SIDE.production_gate2;
check('held-out fixture aptos_heldout_d1f1ea894da1 (256x256 sample)',
  run(new Uint8ClampedArray(DUMP), { width: SIDE.source_width, height: SIDE.source_height, size: SIDE.source_size_bytes }), (r) => [
    ['the dump is the one its sidecar describes', DUMP.length === N * N * 4 && DUMP_HASH === SIDE.dump_sha256],
    ['the dump has pixels near the cut-off, so this case is not vacuous', E.pixels_with_luminance_in_5_to_40 > 0],
    ['passes gates 1 and 2', r.preflightPassed === true && r.failedGate === null],
    ['foreground count equals the figure computed in Python', Math.round(r.gate2.foregroundCoverage * N * N) === E.foreground_pixels
      && near(r.gate2.foregroundCoverage, E.foreground_coverage, 1e-12)],
    ['R/B ratio equals the figure computed in Python', near(r.gate2.redToBlueRatio, E.red_to_blue_ratio, 1e-9)],
    ['red share equals the figure computed in Python', near(r.gate2.redShare, E.red_share, 1e-9)],
    ['not flagged as a document', r.gate2.isDocumentOrDiagram === false],
    ['the production gate passes the same pixels', P.passed === true],
    ['coverage equals the production gate\'s', near(r.gate2.foregroundCoverage, P.foreground_coverage, 1e-12)],
    ['R/B ratio equals the production gate\'s (float32 means there: 1e-5)', near(r.gate2.redToBlueRatio, P.red_to_blue_ratio, 1e-5)],
    ['red share equals the production gate\'s, which is rounded to 3 decimals', near(r.gate2.redShare, P.red_share_rounded_3dp, 5.0001e-4)],
  ]);

// ---- nothing is reported without a canvas
check('no canvas available', run(DISC_REDDISH, { canvas: false }), (r) => [
  ['cannot pass', failsGate(r, 2)],
  ['coverage not evaluated', r.gate2.coverageEvaluated === false],
  ['metric says not evaluated', /Not evaluated in the browser/.test(r.gate2.metric)],
  ['no colour figure and no focus estimate', r.gate2.redToBlueRatio === null && r.gate2.redShare === null && r.gate3.focusEstimate === null],
]);

// The count is what ran, not a constant.
console.log(`PREFLIGHT CHECK: ${ran - failures.length}/${ran} cases as expected`);
process.exit(failures.length ? 1 : 0);
