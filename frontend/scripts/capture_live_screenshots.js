/**
 * Capture the Chapter 4 interface figures against the RUNNING STACK.
 *
 * capture_screenshots.js drives the frontend's built-in demo fixtures with no
 * backend attached. That photographs layout deterministically, but every score,
 * Grad-CAM overlay and gate metric in the resulting figures is a fixture, which
 * is why the manifest has to disclaim them in a paragraph.
 *
 * This script signs in to the real API, uploads a real fundus photograph, and
 * waits for the real admission gates and the real digest-verified EfficientNet-B0
 * checkpoint to answer. What appears in these figures is what the system produced.
 *
 * It REFUSES to run rather than fall back to fixtures, because a fixture-driven
 * capture is visually indistinguishable from a real one.
 *
 * Usage:
 *   node scripts/capture_live_screenshots.js
 *   IMAGE=/path/to/fundus.jpg node scripts/capture_live_screenshots.js
 */

import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME = process.env.CHROME_PATH
  || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const BASE = process.env.BASE_URL || 'http://127.0.0.1:3000';
const API = process.env.API_ORIGIN || 'http://127.0.0.1:8000';
const IMAGE = process.env.IMAGE
  || 'C:\\Users\\USER\\Downloads\\fundus image 2b.jpg';

const USER = process.env.CDSS_USER || 'demo.clinician';
const PASS = process.env.CDSS_PASSWORD || 'dr_secure_password_2026';

const OUT = path.resolve('../docs/chapter4/screenshots_live');

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const shot = (page, name) => page.screenshot({ path: path.join(OUT, name) });

async function clickText(page, text, tags = 'button, a') {
  const handle = await page.evaluateHandle((t, sel) => {
    const el = [...document.querySelectorAll(sel)].find(
      (e) => e.textContent.trim().toLowerCase().includes(t.toLowerCase()));
    return el || null;
  }, text, tags);
  const el = handle.asElement();
  if (!el) return false;
  await el.click();
  return true;
}

async function preflight() {
  const problems = [];
  try {
    const res = await fetch(`${API}/health`);
    const body = await res.json();
    if (body.inference_ready !== true) {
      problems.push(
        `${API}/health reports inference_ready=${body.inference_ready}. `
        + `Detail: ${String(body.inference_detail).slice(0, 160)}`);
    }
    if (String(body.inference_engine).toLowerCase().includes('mock')) {
      problems.push('The backend is serving the SIMULATED engine. These figures '
        + 'must show the evaluated checkpoint.');
    }
  } catch (err) {
    problems.push(`No backend at ${API}/health (${err.message})`);
  }

  try {
    const r = await fetch(BASE);
    if (!r.ok) problems.push(`${BASE} returned HTTP ${r.status}`);
  } catch (err) {
    problems.push(`No frontend at ${BASE} (${err.message})`);
  }

  if (!fs.existsSync(IMAGE)) problems.push(`No image at ${IMAGE}`);

  if (problems.length) {
    console.error('\nREFUSING TO CAPTURE\n');
    problems.forEach((p) => console.error(`  - ${p}`));
    console.error('\nFalling back to the frontend fixtures would produce figures'
      + '\nthat look identical and evidence nothing. Fix the above and re-run.\n');
    process.exit(1);
  }
}

async function main() {
  await preflight();
  fs.mkdirSync(OUT, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
    defaultViewport: { width: 1440, height: 960, deviceScaleFactor: 2 },
  });
  const page = await browser.newPage();

  const apiCalls = [];
  page.on('response', (r) => {
    if (r.url().includes('/api/v1/')) apiCalls.push(`${r.status()} ${r.url().split('/api/v1')[1]}`);
  });

  const captured = [];
  const snap = async (name, label) => {
    await shot(page, name);
    captured.push(name);
    console.log(`  ${name.padEnd(42)} ${label}`);
  };

  // 1 — sign-in
  await page.goto(`${BASE}/#signin`, { waitUntil: 'networkidle2', timeout: 30000 });
  await page.waitForSelector('#staff-id', { timeout: 20000 });
  await page.type('#staff-id', USER);
  await page.type('#password', PASS);
  await wait(500);
  await snap('01_signin_screen.png', 'credentials entered, not yet submitted');

  await page.click('button[type="submit"]');
  await page.waitForFunction(() => !window.location.hash.includes('signin'),
    { timeout: 30000 });
  await wait(2500);

  // 2 — dashboard
  await snap('02_clinical_dashboard.png', 'live worklist from the API');

  // 3 — new assessment with a real image
  if (!await clickText(page, 'New Assessment')) {
    await page.goto(`${BASE}/#new_assessment`, { waitUntil: 'networkidle2' });
  }
  await wait(2000);

  const patient = await page.$('input[placeholder*="PT-"]');
  if (patient) await patient.type(`PT-${Date.now().toString().slice(-5)}`);

  const fileInput = await page.$('input[type="file"]');
  if (!fileInput) throw new Error('No file input on the assessment screen');
  await fileInput.uploadFile(IMAGE);
  await wait(2500);
  await snap('03_new_assessment_upload.png', 'real fundus photograph selected');

  // 4 — validation stepper
  await clickText(page, 'Proceed to 3-Gate Validation');
  await wait(3500);
  await snap('04_validation_stepper_passed.png', 'three gates, real metrics');
  await wait(6000);

  // 5 — decision support
  await snap('05_decision_support_workspace.png', 'real grade + real Grad-CAM');

  // 6 — review modal
  if (await clickText(page, 'Review') || await clickText(page, 'Certify')) {
    await wait(2000);
    await snap('06_professional_review_modal.png', 'human-in-the-loop modal');
    await page.keyboard.press('Escape');
    await wait(1000);
  }

  // 7 — record history
  if (await clickText(page, 'Record History')) {
    await wait(2500);
    await snap('08_record_history_audit.png', 'live record history');
  }

  await browser.close();

  console.log(`\n${captured.length} figure(s) -> ${OUT}`);
  console.log('\nAPI calls observed during capture:');
  const counts = apiCalls.reduce((a, c) => (a[c] = (a[c] || 0) + 1, a), {});
  Object.entries(counts).forEach(([k, v]) => console.log(`  ${String(v).padStart(3)}x  ${k}`));

  const failures = apiCalls.filter((c) => !/^2\d\d/.test(c));
  if (failures.length) {
    console.log('\nNON-2xx RESPONSES during capture:');
    [...new Set(failures)].forEach((f) => console.log(`  ${f}`));
    console.log('Figures captured while the API was failing are not evidence.');
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Capture failed:', err.message);
  process.exit(1);
});
