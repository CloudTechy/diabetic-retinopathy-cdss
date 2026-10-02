/**
 * Capture the fail-closed outcome from a surface that DURABLY shows it.
 *
 * The validation stepper replays its animation when a stored assessment is
 * reopened, and the rejection frame passes in under a second - two attempts to
 * photograph it produced a figure of Gate 1 still spinning, and the capture
 * correctly refused to ship either.
 *
 * The worklist filtered to rejected records shows the same outcome and keeps
 * showing it, so it is the honest surface for this figure. The measured
 * evidence for the fail-closed path is validation_test_results.csv, where six
 * derived negatives are each rejected with their error code; this figure
 * evidences that the interface surfaces the refusal to the clinician.
 */
import puppeteer from 'puppeteer-core';
import path from 'path';

const CHROME = process.env.CHROME_PATH
  || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const BASE = 'http://127.0.0.1:3000';
const OUT = path.resolve('../docs/chapter4/screenshots_live');
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
  defaultViewport: { width: 1440, height: 1100, deviceScaleFactor: 2 },
});
const page = await browser.newPage();

await page.goto(`${BASE}/#signin`, { waitUntil: 'domcontentloaded' });
await page.waitForFunction(
  () => document.querySelector('#staff-id')
    || /Worklist|New Assessment/.test(document.body.innerText),
  { timeout: 60000 });
await page.evaluate(() => {
  const b = [...document.querySelectorAll('button')]
    .find((x) => /Dr\. Demo|Demo \(Ophth\)/i.test(x.textContent || ''));
  if (b) b.click();
});
await wait(800);
await page.click('button[type="submit"]');
await page.waitForFunction(
  () => /Worklist|New Assessment/.test(document.body.innerText), { timeout: 30000 });
await wait(2000);

// Filter the worklist to the refused studies.
const filtered = await page.evaluate(() => {
  const sel = [...document.querySelectorAll('select')]
    .find((s) => [...s.options].some((o) => /reject/i.test(o.value || o.textContent)));
  if (!sel) return false;
  const opt = [...sel.options].find((o) => /reject/i.test(o.value || o.textContent));
  sel.value = opt.value;
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
});

if (!filtered) {
  await browser.close();
  throw new Error('No rejected-status filter on the worklist.');
}

await wait(2500);

const shows = await page.evaluate(() => {
  const t = document.body.innerText;
  const m = t.match(/ERR_[A-Z_]+|Rejected|Technical Rejection/i);
  return m ? m[0] : null;
});

if (!shows) {
  await browser.close();
  throw new Error('The filtered worklist does not show a rejected study.');
}

await page.screenshot({
  path: path.join(OUT, '04b_validation_stepper_rejected.png'),
});
console.log(`  04b_validation_stepper_rejected.png        worklist filtered to refused studies — shows ${JSON.stringify(shows)}`);

await browser.close();
