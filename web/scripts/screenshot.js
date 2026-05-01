const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 2,
  });
  const page = await ctx.newPage();

  page.on("console", (msg) => console.log("[browser]", msg.type(), msg.text()));
  page.on("pageerror", (err) => console.log("[pageerror]", err.message));

  const url = process.env.URL || "http://127.0.0.1:7300/";
  console.log("navigating to", url);
  await page.goto(url, { waitUntil: "networkidle", timeout: 120_000 });

  // Wait for graph to render. Sigma puts a <canvas> inside the container.
  // We poll until at least one canvas exists and has non-trivial pixels.
  await page.waitForSelector("canvas", { timeout: 60_000 });
  console.log("canvas exists, waiting for sigma to settle...");
  await page.waitForTimeout(12_000);

  const out = process.env.OUT || "/tmp/voice-twin.png";
  await page.screenshot({ path: out, fullPage: false });
  console.log("saved", out);

  await browser.close();
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
