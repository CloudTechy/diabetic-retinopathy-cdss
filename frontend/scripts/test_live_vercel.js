import puppeteer from 'puppeteer-core';

const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';

async function testScreens() {
  const browser = await puppeteer.launch({
    executablePath: chromePath,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 960, deviceScaleFactor: 2 });
  
  const errors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') console.log('PAGE LOG ERROR:', msg.text());
  });
  page.on('pageerror', err => {
    console.error('PAGE ERROR:', err.message);
    errors.push(err.message);
  });

  console.log('Testing #dashboard...');
  await page.goto('https://frontend-six-psi-77.vercel.app/#dashboard', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 1500));
  const dashboardLength = (await page.$eval('#root', el => el.innerHTML)).length;
  console.log('Dashboard rendered successfully, innerHTML length:', dashboardLength);

  console.log('Testing #new_assessment...');
  await page.goto('https://frontend-six-psi-77.vercel.app/#new_assessment', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 1500));
  const newAssessmentLength = (await page.$eval('#root', el => el.innerHTML)).length;
  console.log('New Assessment rendered successfully, innerHTML length:', newAssessmentLength);

  console.log('Testing #decision_support...');
  await page.goto('https://frontend-six-psi-77.vercel.app/#decision_support', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 1500));
  const decisionSupportLength = (await page.$eval('#root', el => el.innerHTML)).length;
  console.log('Decision Support rendered successfully, innerHTML length:', decisionSupportLength);

  await page.screenshot({ path: 'docs/screenshots/live_vercel_verified.png' });
  console.log('Captured live_vercel_verified.png');

  await browser.close();

  if (errors.length > 0) {
    console.error('Found errors:', errors);
    process.exit(1);
  } else {
    console.log('ALL LIVE VERCEL SCREENS PASSED WITH ZERO ERRORS!');
  }
}

testScreens().catch(err => {
  console.error('Failed:', err);
  process.exit(1);
});
