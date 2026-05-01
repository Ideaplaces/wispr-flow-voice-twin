// Capture three states of the explorer: default, node selected, topic filtered.
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 2,
  });
  const page = await ctx.newPage();
  page.on("pageerror", (err) => console.log("[pageerror]", err.message));

  await page.goto("http://127.0.0.1:7300/", { waitUntil: "networkidle", timeout: 120_000 });
  await page.waitForSelector("canvas", { timeout: 60_000 });
  await page.waitForTimeout(12_000);

  // 1. default
  await page.screenshot({ path: "/tmp/voice-twin-default.png" });
  console.log("default saved");

  // 2. click a topic in the sidebar to isolate it.
  // pick the 5th topic for variety
  const topicButtons = page.locator("aside button").filter({ hasNotText: /clear|back|show all/i });
  const count = await topicButtons.count();
  console.log("topic buttons found:", count);
  if (count >= 5) {
    await topicButtons.nth(4).click();
    await page.waitForTimeout(2500);
    await page.screenshot({ path: "/tmp/voice-twin-topic-filter.png" });
    console.log("topic-filter saved");
    // clear filter
    await page.locator("aside button", { hasText: "clear" }).first().click();
    await page.waitForTimeout(1500);
  }

  // 3. Pick a node from the first colored topic and walk pixel-by-pixel from
  //    its UMAP coordinate via sigma's camera transform until we find one
  //    sigma can actually hit-test. We do this through page.evaluate so we
  //    don't have to re-derive the camera transform here.
  const result = await page.evaluate(async () => {
    // Find a candidate node id from a top topic (color != #1a1a1a / #3a3a3a).
    const res = await fetch("/api/graph");
    const data = await res.json();
    const topNode = data.nodes.find(
      (n) => n.color !== "#1a1a1a" && n.color !== "#3a3a3a" && n.topic_id !== -1
    );
    if (!topNode) return { ok: false, reason: "no labeled node" };

    // Locate sigma instance the React side dropped on the window, fall back
    // to walking the canvas DOM and using event simulation if absent.
    // Easier path: dispatch a synthetic clickNode event on the renderer if
    // we can reach it; otherwise we just return the node id and let the
    // host script click on its pixel coordinates.
    const sigmaEls = document.querySelectorAll("canvas");
    const canvas = sigmaEls[sigmaEls.length - 1];
    const rect = canvas.getBoundingClientRect();
    return {
      ok: true,
      id: topNode.id,
      label: topNode.topic_label,
      rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height },
    };
  });
  console.log("target node:", result);

  if (result.ok) {
    // Reload with ?select=<id> so the React side sets selection on mount.
    await page.goto(`http://127.0.0.1:7300/?select=${result.id}`, {
      waitUntil: "networkidle",
      timeout: 120_000,
    });
    await page.waitForSelector("canvas", { timeout: 60_000 });
    await page.waitForTimeout(12_000);
    await page.screenshot({ path: "/tmp/voice-twin-node-selected.png" });
    console.log("node-selected saved (selected:", result.id, "->", result.label, ")");
  }

  await browser.close();
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
