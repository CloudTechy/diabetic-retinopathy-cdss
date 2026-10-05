/**
 * Open every screen at a phone's width against the RUNNING stack and report
 * anything that makes the page scroll sideways.
 *
 * A page is reported when the document is wider than the viewport; the
 * elements whose right edge crosses the viewport are listed, skipping those
 * inside a container that scrolls horizontally on purpose (tables).
 *
 * Usage:  node scripts/audit_mobile_layout.js [outDir]
 * Exit status 1 if any screen overflows.
 */
import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME = process.env.CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const BASE = process.env.BASE_URL || 'http://127.0.0.1:3000';
const USER = process.env.CDSS_USER || 'demo.clinician';
const PASS = process.env.CDSS_PASSWORD || 'dr_secure_password_2026';
const COMPLETED = process.env.COMPLETED_ID || '';
const REJECTED = process.env.REJECTED_ID || '';
const OUT = process.argv[2] ? path.resolve(process.argv[2]) : null;
const WIDTH = Number(process.env.WIDTH || 360);

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const measure = () => {
  const vw = document.documentElement.clientWidth;
  const scrollsOnPurpose = (el) => {
    for (let p = el.parentElement; p; p = p.parentElement) {
      const ox = getComputedStyle(p).overflowX;
      if ((ox === 'auto' || ox === 'scroll' || ox === 'hidden') && p !== document.body && p !== document.documentElement) return true;
    }
    return false;
  };
  const offenders = [];
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    if (r.right > vw + 1 && !scrollsOnPurpose(el)) {
      offenders.push(`${el.tagName.toLowerCase()}.${String(el.className).split(' ').slice(0, 4).join('.')} right=${Math.round(r.right)} "${(el.textContent || '').trim().slice(0, 40)}"`);
    }
  }
  const hasMobileNav = !!document.querySelector('nav[aria-label="Clinical Navigation (mobile)"]');
  return { vw, docWidth: document.documentElement.scrollWidth, offenders: offenders.slice(0, 8), hasMobileNav };
};

const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new' });
const page = await browser.newPage();
await page.setViewport({ width: WIDTH, height: 780, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
let failed = 0;

async function check(name, hash, { expectNav = true, before } = {}) {
  await page.goto(`${BASE}/#${hash}`, { waitUntil: 'networkidle2', timeout: 30000 });
  await page.reload({ waitUntil: 'networkidle2' });
  await wait(1200);
  if (before) await before();
  const m = await page.evaluate(measure);
  const overflow = m.docWidth > m.vw + 1;
  const navMissing = expectNav && !m.hasMobileNav;
  if (overflow || navMissing) failed += 1;
  console.log(`${overflow || navMissing ? 'FAIL' : 'ok  '} ${name}: document ${m.docWidth}px in a ${m.vw}px viewport${navMissing ? '; NO mobile navigation' : ''}`);
  for (const o of overflow ? m.offenders : []) console.log(`       ${o}`);
  if (OUT) {
    fs.mkdirSync(OUT, { recursive: true });
    await page.screenshot({ path: path.join(OUT, `${name}.png`), fullPage: true });
  }
}

await check('signin', 'signin', { expectNav: false });
await page.evaluate((u, p) => {
  const set = (sel, val) => {
    const el = document.querySelector(sel);
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(el, val);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };
  set('#staff-id', u);
  set('#password', p);
}, USER, PASS);
await page.click('button[type="submit"]');
await wait(2500);

await check('dashboard', 'dashboard');
await check('new_assessment', 'new_assessment');
await check('history', 'history');
if (COMPLETED) {
  await check('completed_assessment', `completed_assessment/${COMPLETED}`);
  await check('decision_support', `decision_support/${COMPLETED}`);
}
if (REJECTED) await check('validation_rejected', `validation/${REJECTED}`);

await browser.close();
console.log(failed ? `MOBILE LAYOUT: ${failed} screen(s) with a problem at ${WIDTH}px` : `MOBILE LAYOUT: every screen fits ${WIDTH}px`);
process.exit(failed ? 1 : 0);
