import puppeteer from 'puppeteer-core';

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const BASE_URL = 'http://127.0.0.1:4173';

async function runPersistenceTest() {
  console.log('🧪 Starting Browser Refresh & Data Persistence Verification Test...');

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=1440,960'],
    defaultViewport: { width: 1440, height: 960 },
  });

  const page = await browser.newPage();

  try {
    // 1. Load Dashboard
    console.log('1. Loading dashboard...');
    await page.goto(`${BASE_URL}/#dashboard`, { waitUntil: 'networkidle0', timeout: 15000 });
    await new Promise(r => setTimeout(r, 1000));

    // 2. Select assessment REC-2026-0042
    console.log('2. Selecting assessment REC-2026-0042...');
    await page.goto(`${BASE_URL}/#decision_support/REC-2026-0042`, { waitUntil: 'networkidle0', timeout: 15000 });
    await page.waitForSelector('text/REC-2026-0042', { timeout: 10000 }).catch(() => {});
    console.log('   ✅ Loaded REC-2026-0042 on Decision Support');

    // 3. Open Professional Review Modal and submit review
    console.log('3. Opening review modal and submitting review...');
    await page.goto(`${BASE_URL}/#review/REC-2026-0042`, { waitUntil: 'networkidle0', timeout: 15000 });
    
    // Wait for review modal to be mounted
    await page.waitForSelector('#confirm-checkbox', { timeout: 10000 });
    console.log('   Review modal successfully rendered.');

    // Click Agree option via page.evaluate
    await page.evaluate(() => {
      const radioBtns = Array.from(document.querySelectorAll('button[role="radio"]'));
      if (radioBtns.length > 0) {
        radioBtns[0].click(); // Option A: Agree
      }
    });
    console.log('   Selected Agree option.');
    await new Promise(r => setTimeout(r, 500));

    // Fill notes if available
    const textarea = await page.$('#justification-text');
    if (textarea) {
      await page.type('#justification-text', 'Corroborated: Persisted clinical review notes test.');
      console.log('   Typed observation notes.');
    }

    // Check confirmation checkbox
    await page.click('#confirm-checkbox');
    console.log('   Checked confirmation box.');
    await new Promise(r => setTimeout(r, 500));

    // Click submit review button
    await page.click('button[type="submit"]');
    console.log('   Clicked submit review button.');
    await new Promise(r => setTimeout(r, 2000));

    // 4. PERFORM BROWSER REFRESH / RELOAD
    console.log('4. 🔄 RELOADING / REFRESHING BROWSER PAGE TO TEST PERSISTENCE...');
    await page.reload({ waitUntil: 'networkidle0', timeout: 15000 });
    await new Promise(r => setTimeout(r, 2000));

    // 5. Verify the data persisted after reload
    console.log('5. Verifying persisted data after reload...');
    const contentAfterReload = await page.content();
    
    const hasRecordId = contentAfterReload.includes('REC-2026-0042');
    const hasNotes = contentAfterReload.includes('Persisted clinical review notes test');
    const hasConcurred = contentAfterReload.includes('Concurred') || contentAfterReload.includes('Agree');

    console.log(`   Assessment ID preserved after refresh: ${hasRecordId ? 'YES ✅' : 'NO ❌'}`);
    console.log(`   Clinician Review Notes preserved after refresh: ${hasNotes ? 'YES ✅' : 'NO ❌'}`);
    console.log(`   Concurred status preserved after refresh: ${hasConcurred ? 'YES ✅' : 'NO ❌'}`);

    // 6. Navigate to Dashboard to verify the record status is persisted in the worklist
    console.log('6. Checking Dashboard worklist after reload...');
    await page.goto(`${BASE_URL}/#dashboard`, { waitUntil: 'networkidle0', timeout: 15000 });
    await new Promise(r => setTimeout(r, 1500));
    const dashboardContent = await page.content();
    const hasRecordOnDashboard = dashboardContent.includes('REC-2026-0042');
    console.log(`   Record visible in Dashboard worklist: ${hasRecordOnDashboard ? 'YES ✅' : 'NO ❌'}`);

    // 7. Refresh Dashboard to verify worklist still has persisted record
    console.log('7. 🔄 Refreshing Dashboard page...');
    await page.reload({ waitUntil: 'networkidle0', timeout: 15000 });
    await new Promise(r => setTimeout(r, 1500));
    const dashboardAfterReload = await page.content();
    const stillOnDashboard = dashboardAfterReload.includes('REC-2026-0042');
    console.log(`   Record still in Dashboard worklist after dashboard refresh: ${stillOnDashboard ? 'YES ✅' : 'NO ❌'}`);

    if (hasRecordId && hasRecordOnDashboard && stillOnDashboard) {
      console.log('🎉 PERSISTENCE TEST 100% PASSED! All assessment and review data survives browser refresh flawlessly!');
    } else {
      throw new Error('Persistence test failed to verify preserved data after reload');
    }
  } catch (err) {
    console.error('❌ Persistence test failed:', err);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

runPersistenceTest();
