import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const BASE_URL = 'http://127.0.0.1:4173';

const DOCS_DIR = path.resolve('docs/screenshots');
const ARTIFACTS_DIR = 'C:\\Users\\USER\\.gemini\\antigravity\\brain\\4c03715b-d77f-4960-9f39-1c9fc499e344\\screenshots';

fs.mkdirSync(DOCS_DIR, { recursive: true });
fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });

const screens = [
  { hash: '#signin', name: '01_signin_screen.png', title: 'Screen 1: Clinical Sign-In & Authentication', waitMs: 1500 },
  { hash: '#dashboard', name: '02_clinical_dashboard.png', title: 'Screen 2: Clinical Dashboard & Triage Worklist', waitMs: 1500 },
  { hash: '#new_assessment', name: '03_new_assessment_upload.png', title: 'Screen 3: Retinal Fundus Image Ingestion & De-identification', waitMs: 1500 },
  { hash: '#validation', name: '04_validation_stepper_passed.png', title: 'Screen 4: Real-Time 3-Stage Technical Quality Stepper (Passed)', waitMs: 4500 },
  { hash: '#rejected', name: '04b_validation_stepper_rejected.png', title: 'Screen 4b: 3-Stage Technical Quality Stepper (Gate 3 Rejection Fail-Closed)', waitMs: 4500 },
  { hash: '#decision_support', name: '05_decision_support_workspace.png', title: 'Screen 5: Decision-Support Workspace (Fundus Viewer & Grad-CAM Blend)', waitMs: 2000 },
  { hash: '#review', name: '06_professional_review_modal.png', title: 'Screen 6: Clinician Professional Review & Certification Modal', waitMs: 2000 },
  { hash: '#completed', name: '07_completed_assessment_record.png', title: 'Screen 7: Immutable Finalized Record & Tamper-Evident Certification', waitMs: 2000 },
  { hash: '#history', name: '08_record_history_audit.png', title: 'Screen 8: Longitudinal Record History & Audit Drawer', waitMs: 2000 },
];

async function capture() {
  console.log('🚀 Launching headless Chrome for UI Smoke Test & Chapter 4 Screenshot Suite...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=1440,960'],
    defaultViewport: { width: 1440, height: 960, deviceScaleFactor: 2 },
  });

  for (const s of screens) {
    const page = await browser.newPage();
    const targetUrl = `${BASE_URL}/${s.hash}`;
    console.log(`📸 Capturing ${s.title} (${targetUrl})...`);
    await page.goto(targetUrl, { waitUntil: 'networkidle0', timeout: 15000 });
    
    // Allow canvas/WebGL or stepper animation to finish
    await new Promise(r => setTimeout(r, s.waitMs));

    const docsPath = path.join(DOCS_DIR, s.name);
    const artifactsPath = path.join(ARTIFACTS_DIR, s.name);

    await page.screenshot({ path: docsPath, fullPage: false });
    fs.copyFileSync(docsPath, artifactsPath);
    console.log(`   ✅ Saved to: ${docsPath}`);
    await page.close();
  }

  await browser.close();
  console.log('🎉 All 9 clinical UI screenshots successfully captured!');
}

capture().catch(err => {
  console.error('❌ Screenshot capture error:', err);
  process.exit(1);
});
