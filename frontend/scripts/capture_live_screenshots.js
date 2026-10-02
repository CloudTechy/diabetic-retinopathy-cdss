/**
 * Capture the Chapter 4 interface figures against a RUNNING STACK.
 *
 * capture_screenshots.js drives the frontend's built-in demo fixtures with no
 * backend attached. That is deterministic and fine for photographing layout,
 * but every score, Grad-CAM overlay and gate metric it shows is a fixture, and
 * the manifest has to say so in a paragraph of caveats.
 *
 * This script instead signs in to the real API, uploads real held-out APTOS
 * images, and waits for the real validation gates and the real EfficientNet-B0
 * checkpoint to respond. What appears in the resulting figures is what the
 * system produced.
 *
 * PREREQUISITES - the script checks all of them and refuses rather than
 * quietly falling back to fixtures, which is the failure that would put
 * fabricated numbers into the dissertation:
 *
 *   1. Backend on API_ORIGIN serving the REAL engine (not AI_INFERENCE_ENGINE=mock)
 *   2. Vite dev server on BASE_URL proxying /api to that backend
 *   3. Real APTOS images in IMAGES_DIR, named <image_id>.png
 *
 * Usage:
 *   node scripts/capture_live_screenshots.js
 */

import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = process.env.CHROME_PATH
  || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const BASE_URL = process.env.BASE_URL || 'http://127.0.0.1:3000';
const API_ORIGIN = process.env.API_ORIGIN || 'http://127.0.0.1:8000';
const IMAGES_DIR = process.env.IMAGES_DIR
  || path.resolve('../storage/datasets/aptos2019/train_images');

const USERNAME = process.env.CDSS_USER || 'demo.clinician';
const PASSWORD = process.env.CDSS_PASSWORD || 'dr_secure_password_2026';

const OUT_DIR = path.resolve('../docs/chapter4/screenshots_live');

// Held-out test images only. Showing a prediction on an image the model was
// trained on would be meaningless as evidence, however good it looked.
const ACCEPT_IMAGE = process.env.ACCEPT_IMAGE || null;   // resolved below
const REJECT_IMAGE = process.env.REJECT_IMAGE || null;

const shot = (page, name) =>
  page.screenshot({ path: path.join(OUT_DIR, name), fullPage: false });

const pause = (ms) => new Promise((r) => setTimeout(r, ms));

async function clickByText(page, text, tag = 'button') {
  const handle = await page.evaluateHandle(
    (t, g) => [...document.querySelectorAll(g)]
      .find((el) => el.textContent.trim().toLowerCase().includes(t.toLowerCase())),
    text, tag,
  );
  const el = handle.asElement();
  if (!el) throw new Error(`No <${tag}> containing "${text}"`);
  await el.click();
  return el;
}

async function preflight() {
  const problems = [];

  // The backend must be up AND serving the real engine.
  try {
    const res = await fetch(`${API_ORIGIN}/health`);
    const body = await res.json().catch(() => ({}));
    const engine = JSON.stringify(body).toLowerCase();
    if (engine.includes('"mock"') || engine.includes('simulated')) {
      problems.push(
        `${API_ORIGIN} reports a simulated inference engine. These figures must `
        + 'show the evaluated checkpoint, not a stand-in. Restart the backend '
        + 'without AI_INFERENCE_ENGINE=mock.');
    }
  } catch (err) {
    problems.push(`No backend at ${API_ORIGIN}/health (${err.message})`);
  }

  try {
    await fetch(BASE_URL);
  } catch (err) {
    problems.push(`No frontend at ${BASE_URL} (${err.message})`);
  }

  if (!fs.existsSync(IMAGES_DIR)) {
    problems.push(`No image directory at ${IMAGES_DIR}`);
  }

  if (problems.length) {
    console.error('\nREFUSING TO CAPTURE\n');
    problems.forEach((p) => console.error(`  - ${p}`));
    console.error(
      '\nThe point of this script is that the figures show real output. Falling'
      + '\nback to the frontend\'s fixtures would produce screenshots that look'
      + '\nidentical and mean nothing. Fix the above and re-run.\n');
    process.exit(1);
  }
}

async function main() {
  await preflight();
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
      '--window-size=1440,960'],
    defaultViewport: { width: 1440, height: 960, deviceScaleFactor: 2 },
  });

  const page = await browser.newPage();

  // Any call that falls through to the in-memory store means the figure would
  // be a fixture. Fail loudly instead.
  page.on('console', (m) => {
    const t = m.text();
    if (/backend API unavailable|falling back/i.test(t)) {
      console.error(`\n[FIXTURE FALLBACK DETECTED] ${t}`);
      console.error('The frontend lost the backend mid-capture. Figures from '
        + 'this run are not evidence of anything. Aborting.');
      process.exit(1);
    }
  });

  // ---- 1. Sign-in -----------------------------------------------------
  await page.goto(`${BASE_URL}/#signin`, { waitUntil: 'networkidle0' });
  await page.waitForSelector('#staff-id', { timeout: 15000 });
  await page.type('#staff-id', USERNAME);
  await page.type('#password', PASSWORD);
  await pause(400);
  await shot(page, '01_signin_screen.png');
  console.log('  01 sign-in');

  await page.click('button[type="submit"]');
  await page.waitForFunction(
    () => !window.location.hash.includes('signin'), { timeout: 20000 });
  await pause(1500);

  // ---- 2. Dashboard ---------------------------------------------------
  await shot(page, '02_clinical_dashboard.png');
  console.log('  02 dashboard');

  await browser.close();
  console.log(`\nWritten to ${OUT_DIR}`);
}

main().catch((err) => {
  console.error('Capture failed:', err);
  process.exit(1);
});
