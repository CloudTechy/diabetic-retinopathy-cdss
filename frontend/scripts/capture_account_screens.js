/**
 * Capture the account-administration figures against the RUNNING STACK.
 *
 * Nothing here is a fixture and nothing is seeded: the script needs a Super
 * Administrator that already exists (created by
 * backend/scripts/bootstrap_superadmin.py or the SUPERADMIN_* environment)
 * and its CURRENT password, given through the environment. It then performs
 * the defence sequence for real - registration request, approval, physician
 * creation with a temporary password, first sign-in password change, clean
 * physician dashboard - and photographs each state only after observing it
 * on screen. It refuses to write a frame whose captioned state it cannot see
 * and exits non-zero on any failure. The temporary password it reads from
 * the screen is used once, to sign the physician in, and is never printed.
 *
 * Usage:
 *   CDSS_ADMIN_USER=root.admin CDSS_ADMIN_PASSWORD='...' node scripts/capture_account_screens.js
 *   (optional: BASE_URL, API_ORIGIN, CHROME_PATH, OUT_DIR, RUN_ID)
 */
import puppeteer from 'puppeteer-core';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

const CHROME = process.env.CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const BASE = process.env.BASE_URL || 'http://127.0.0.1:3000';
const API = process.env.API_ORIGIN || 'http://127.0.0.1:8000';
const ADMIN_USER = process.env.CDSS_ADMIN_USER;
const ADMIN_PASS = process.env.CDSS_ADMIN_PASSWORD;
const OUT = path.resolve(process.env.OUT_DIR || '../docs/chapter4/screenshots/accounts');
const RUN = process.env.RUN_ID || new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');

if (!ADMIN_USER || !ADMIN_PASS) {
  console.error('CDSS_ADMIN_USER and CDSS_ADMIN_PASSWORD are required (an existing Super Administrator); nothing is seeded here.');
  process.exit(2);
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (m) => console.log(`[accounts] ${m}`);
const esc = (s) => s.replace(/\./g, '\\.');

// Passwords are drawn fresh for every run and never written anywhere: the
// accounts this script creates must not share a credential that sits in the
// repository. The prefix guarantees the policy's letter and digit.
const runPassword = () => `Pw7-${crypto.randomBytes(12).toString('hex')}`;
const REGISTRANT = { username: `applicant.${RUN}`, fullName: 'Registration Applicant (demonstration)', password: runPassword() };
const PHYSICIAN = { username: `physician.${RUN}`, fullName: 'Demonstration Physician', newPassword: runPassword() };

async function api(method, route, body, token) {
  const res = await fetch(`${API}/api/v1${route}`, {
    method,
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${method} ${route} -> ${res.status} ${await res.text()}`);
  return res.status === 204 ? null : res.json();
}

// React controlled inputs: set through the native setter so onChange fires.
const setValue = async (page, selector, value) => {
  await page.waitForSelector(selector, { timeout: 60000 });
  await page.evaluate((sel, val) => {
    const el = document.querySelector(sel);
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(el, val);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, selector, value);
};

const bodyText = (page) => page.evaluate(() => document.body.innerText);

async function requireState(page, present, absent, what) {
  // case-insensitive: innerText carries the CSS text-transform of upper-cased headings
  const want = new RegExp(present.source, 'i');
  const avoid = absent ? new RegExp(absent.source, 'i') : null;
  for (let i = 0; i < 240; i++) {
    const text = await bodyText(page);
    if (want.test(text) && !(avoid && avoid.test(text))) return;
    await wait(250);
  }
  const seen = (await bodyText(page)).replace(/\s+/g, ' ').slice(0, 400);
  throw new Error(`state not observed: ${what} (expected /${present.source}/${absent ? ` without /${absent.source}/` : ''} on screen); the screen showed: ${seen}`);
}

async function shoot(page, name, present, what, absent = /Loading|Filtering records|Authenticating/) {
  await requireState(page, present, absent, what);
  await wait(500);
  await page.screenshot({ path: path.join(OUT, `${name}.png`), fullPage: true });
  log(`captured ${name}`);
}

async function goto(page, hash) {
  await page.goto(`${BASE}/#${hash}`, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForSelector('#root *', { timeout: 120000 });
}

async function freshSession(page) {
  await goto(page, 'signin');
  await page.evaluate(() => { localStorage.clear(); sessionStorage.clear(); });
  await goto(page, 'signin');
  await page.waitForSelector('#staff-id', { timeout: 120000 });
}

async function signIn(page, username, password) {
  await freshSession(page);
  await setValue(page, '#staff-id', username);
  await setValue(page, '#password', password);
  await page.click('button[type="submit"]');
}

async function openTab(page, label) {
  await page.evaluate((l) => {
    const btn = [...document.querySelectorAll('nav[aria-label="Console sections"] button')].find((b) => b.textContent.includes(l));
    if (!btn) throw new Error(`console tab not found: ${l}`);
    btn.click();
  }, label);
}

fs.mkdirSync(OUT, { recursive: true });
const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new' });
const page = await browser.newPage();
await page.setViewport({ width: 1366, height: 860, deviceScaleFactor: 1.5 });

try {
  // 1. sign-in screen: empty form, registration option
  await freshSession(page);
  await shoot(page, '01_sign_in', /Request an account/, 'sign-in screen with registration option');

  // 2. registration screen, filled
  await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => /Request an account/.test(b.textContent)).click());
  await requireState(page, /Request a Physician Account/, null, 'registration screen');
  await setValue(page, '#reg-fullName', REGISTRANT.fullName);
  await setValue(page, '#reg-username', REGISTRANT.username);
  await setValue(page, '#reg-email', `${REGISTRANT.username}@example.org`);
  await setValue(page, '#reg-password', REGISTRANT.password);
  await setValue(page, '#reg-passwordConfirmation', REGISTRANT.password);
  await shoot(page, '02_physician_registration', /Request a Physician Account/, 'registration form filled');

  // 3. pending-approval response
  await page.click('button[type="submit"]');
  await shoot(page, '03_pending_approval_response', /submitted for administrative approval/, 'pending-approval response');

  // a pending account is refused at sign-in, in words (a check, not a frame)
  await signIn(page, REGISTRANT.username, REGISTRANT.password);
  await requireState(page, /awaiting administrative approval/, null, 'pending account refused at sign-in');

  // 4. super administrator console, with the data loaded (the pending request is listed)
  await signIn(page, ADMIN_USER, ADMIN_PASS);
  await shoot(page, '04_superadmin_dashboard',
    new RegExp(`Account Administration[\\s\\S]*Pending registration requests[\\s\\S]*${esc(REGISTRANT.username)}`),
    'super administrator console with the pending request listed');

  // 5. create-physician form, filled
  await openTab(page, 'Create physician');
  await requireState(page, /Create a physician account/, null, 'create-physician tab');
  await setValue(page, '#create-fullName', PHYSICIAN.fullName);
  await setValue(page, '#create-username', PHYSICIAN.username);
  await setValue(page, '#create-email', `${PHYSICIAN.username}@example.org`);
  await setValue(page, '#create-facility', 'Defence demonstration');
  await shoot(page, '05_create_physician_form', /Create a physician account/, 'create-physician form filled');
  // submit; the temporary password is shown once, in the banner
  await page.evaluate(() => [...document.querySelectorAll('form button[type="submit"]')].find((b) => /Create account/.test(b.textContent)).click());
  await requireState(page, new RegExp(`Temporary password[\\s\\S]*${esc(PHYSICIAN.username)}`), null, 'temporary password shown once');
  const temporary = await page.evaluate((u) => {
    const li = [...document.querySelectorAll('li')].find((el) => el.textContent.includes(u) && el.textContent.includes('→'));
    return li ? li.textContent.split('→')[1].trim() : null;
  }, PHYSICIAN.username);
  if (!temporary) throw new Error('temporary password not readable from the banner');
  log(`temporary password issued for ${PHYSICIAN.username} (not printed)`);

  // 6. pending-user approval dialogue
  await openTab(page, 'Overview');
  await requireState(page, new RegExp(esc(REGISTRANT.username)), null, 'pending request listed');
  await page.evaluate((u) => {
    const li = [...document.querySelectorAll('li')].find((el) => el.textContent.includes(u));
    [...li.querySelectorAll('button')].find((b) => /Approve/.test(b.textContent)).click();
  }, REGISTRANT.username);
  await shoot(page, '06_pending_user_approval', /Approve account[\s\S]*Confirm/, 'approval confirmation dialogue');
  await page.evaluate(() => [...document.querySelectorAll('[role="alertdialog"] button')].find((b) => /Confirm/.test(b.textContent)).click());
  await requireState(page, /Approve account: done/, null, 'approval confirmed');

  // 9. account-management table, both accounts present
  await openTab(page, 'All accounts');
  const both = `${esc(PHYSICIAN.username)}[\\s\\S]*${esc(REGISTRANT.username)}|${esc(REGISTRANT.username)}[\\s\\S]*${esc(PHYSICIAN.username)}`;
  await shoot(page, '09_account_management_table', new RegExp(both), 'account table with both accounts');

  // 10. audit history with the events just produced
  await openTab(page, 'Audit history');
  await shoot(page, '10_account_audit_history', /ACCOUNT_APPROVED[\s\S]*PHYSICIAN_ACCOUNT_CREATED|PHYSICIAN_ACCOUNT_CREATED[\s\S]*ACCOUNT_APPROVED/, 'audit history with the events just produced');

  // 7. first sign-in: the physician with the temporary password sees only the password change
  await signIn(page, PHYSICIAN.username, temporary);
  await shoot(page, '07_first_sign_in_password_change', /Set Your Password[\s\S]*temporary/, 'mandatory password change');
  await setValue(page, '#current-password', temporary);
  await setValue(page, '#new-password', PHYSICIAN.newPassword);
  await setValue(page, '#confirm-password', PHYSICIAN.newPassword);
  await page.click('button[type="submit"]');
  await requireState(page, /Sign in again with your new password|Practitioner Sign In/, null, 'signed out after the change');

  // 8. clean physician dashboard (worklist loaded and empty)
  await signIn(page, PHYSICIAN.username, PHYSICIAN.newPassword);
  await shoot(page, '08_clean_physician_dashboard', /No assessment records found/, 'empty worklist of the new physician');
  const session = await api('POST', '/auth/login', { username: PHYSICIAN.username, password: PHYSICIAN.newPassword });
  const worklist = await api('GET', '/assessments', null, session.access_token);
  if (worklist.length !== 0) throw new Error(`the new physician's worklist is not empty: ${worklist.length}`);

  fs.writeFileSync(path.join(OUT, 'capture_record.json'), JSON.stringify({
    capturedAt: new Date().toISOString(), base: BASE, api: API, runId: RUN,
    accountsCreated: [REGISTRANT.username, PHYSICIAN.username],
    note: 'Both accounts were created by this capture against the running stack; the temporary password visible in frame 05 was changed in frame 07 and is no longer valid.',
  }, null, 2) + '\n');
  log(`done: 10 frames in ${OUT}`);
} finally {
  await browser.close();
}
