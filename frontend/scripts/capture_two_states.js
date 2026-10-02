/**
 * Capture the two figures whose UI states cannot be reached reliably by
 * driving the form: a rejected assessment (4b) and a finalised record (7).
 *
 * Both are REAL records already produced by the backend - the rejection came
 * from Gate 3 refusing a deliberately blurred image (ERR_MOTION_OR_DEFOCUS_BLUR,
 * Laplacian 1.50 against a threshold of 4.3), and the completed record carries a
 * clinician's own signed review. This navigates the real interface to them
 * rather than re-driving a multi-step form whose intermediate wording the
 * capture has to guess at.
 *
 * It verifies the page is showing the state it claims before writing anything.
 */
import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME = process.env.CHROME_PATH
  || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const BASE = 'http://127.0.0.1:3000';
const API = 'http://127.0.0.1:8000/api/v1';
const OUT = path.resolve('../docs/chapter4/screenshots_live');

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function api(pathname, token) {
  const res = await fetch(API + pathname, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  return res.json();
}

const login = await fetch(`${API}/auth/login`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username: 'demo.clinician', password: 'dr_secure_password_2026' }),
}).then((r) => r.json());

const token = login.access_token;
const records = await api('/assessments', token);

const rejected = records.find((r) => r.status === 'rejected');
const completed = records.find((r) => r.status === 'completed');

if (!rejected) throw new Error('No rejected assessment exists to photograph.');
if (!completed) throw new Error('No completed assessment exists to photograph.');

console.log(`  rejected record : ${rejected.id}`);
console.log(`  completed record: ${completed.id}`);

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
  defaultViewport: { width: 1440, height: 1100, deviceScaleFactor: 2 },
});
const page = await browser.newPage();

// Sign in through the application's own control so the session is genuine.
await page.goto(`${BASE}/#signin`, { waitUntil: 'networkidle2' });
await page.waitForSelector('#staff-id');
await page.evaluate(() => {
  const b = [...document.querySelectorAll('button')]
    .find((x) => /Dr\. Demo|Demo \(Ophth\)/i.test(x.textContent || ''));
  if (b) b.click();
});
await wait(800);
await page.click('button[type="submit"]');
await page.waitForFunction(
  () => /Worklist|New Assessment/.test(document.body.innerText), { timeout: 30000 });
await wait(1500);

async function capture(hash, file, mustMatch, label) {
  await page.goto(`${BASE}/${hash}`, { waitUntil: 'networkidle2' });
  await page.waitForFunction(
    (re) => new RegExp(re, 'i').test(document.body.innerText),
    { timeout: 45000 }, mustMatch.source);
  await wait(2500);
  await page.screenshot({ path: path.join(OUT, file) });
  const shown = await page.evaluate((re) => {
    const m = document.body.innerText.match(new RegExp(re, 'i'));
    return m ? m[0] : null;
  }, mustMatch.source);
  console.log(`  ${file.padEnd(42)} ${label} — matched ${JSON.stringify(shown)}`);
}

await capture(`#validation/${rejected.id}`,
  '04b_validation_stepper_rejected.png',
  /ERR_[A-Z_]+|Rejected|Recapture|not be graded/,
  'fail-closed rejection');

await capture(`#completed_assessment/${completed.id}`,
  '07_completed_assessment_record.png',
  /Professional Review Response|Integrity ID|SIGNED/,
  'finalised signed record');

await browser.close();
console.log(`\n2 figure(s) -> ${OUT}`);
