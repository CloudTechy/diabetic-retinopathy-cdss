/** Diagnostic: watch what the browser actually does during sign-in. */
import puppeteer from 'puppeteer-core';

const CHROME = process.env.CHROME_PATH
  || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const BASE = 'http://127.0.0.1:3000';

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu'],
  defaultViewport: { width: 1280, height: 900 },
});
const page = await browser.newPage();

page.on('console', (m) => console.log(`  [console.${m.type()}] ${m.text().slice(0, 200)}`));
page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message.slice(0, 200)}`));
page.on('requestfailed', (r) => console.log(`  [reqfailed] ${r.url()} ${r.failure()?.errorText}`));
page.on('response', async (r) => {
  if (!r.url().includes('/auth/login')) return;
  let body = '';
  try { body = (await r.text()).slice(0, 300); } catch { body = '<unreadable>'; }
  console.log(`  [response] ${r.status()} ${r.url()}`);
  console.log(`  [body] ${body}`);
});

await page.goto(`${BASE}/#signin`, { waitUntil: 'networkidle2' });
await page.waitForSelector('#staff-id');

// Use the app's own demo-fill so React state is set the way the UI sets it.
const filled = await page.evaluate(() => {
  const btn = [...document.querySelectorAll('button')]
    .find((b) => /Dr\. Demo|Demo \(Ophth\)/i.test(b.textContent));
  if (btn) { btn.click(); return true; }
  return false;
});
console.log(`  demo-fill button clicked: ${filled}`);
await new Promise((r) => setTimeout(r, 800));

const values = await page.evaluate(() => ({
  id: document.querySelector('#staff-id')?.value,
  pw: document.querySelector('#password')?.value ? '(set)' : '(empty)',
}));
console.log(`  field values: ${JSON.stringify(values)}`);

await page.click('button[type="submit"]');
await new Promise((r) => setTimeout(r, 6000));

const after = await page.evaluate(() => ({
  hash: location.hash,
  err: [...document.querySelectorAll('div,p,span')]
    .map((e) => e.textContent)
    .find((t) => t && t.includes('Authentication failed')) || null,
  token: (() => { try { return localStorage.getItem('access_token') ? 'present' : 'absent'; } catch { return 'blocked'; } })(),
}));
console.log(`  after submit: ${JSON.stringify(after)}`);

await browser.close();
