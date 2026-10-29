// Time Pyodide in a browser: the REPL page running the benchmark's queries
// (``queries``) or the CPU benchmark of micro.py (``micro``).
//
//   node browser.js queries BROWSER SITE_DIR QUERIES.json OUT.json [REPEATS]
//   node browser.js micro BROWSER OUT.json
//
// BROWSER is chromium, firefox or webkit (Playwright's). SITE_DIR is the
// directory of the page with its bundle (see README.md). The page is served
// from a server that lives as long as this script.
const fs = require("fs");
const http = require("http");
const path = require("path");
const { chromium, firefox, webkit } = require("playwright");

const BROWSERS = { chromium, firefox, webkit };
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".zip": "application/zip", ".whl": "application/octet-stream" };

function serve(directory) {
  const server = http.createServer((request, response) => {
    const file = path.join(directory, decodeURIComponent(request.url.split("?")[0]).replace(/^\/$/, "/index.html"));
    if (!file.startsWith(directory) || !fs.existsSync(file)) return response.writeHead(404).end();
    response.writeHead(200, { "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream", "Cache-Control": "no-store" });
    fs.createReadStream(file).pipe(response);
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

async function micro(browserName, out) {
  const browser = await BROWSERS[browserName].launch();
  const page = await browser.newPage();
  await page.goto("about:blank");
  await page.addScriptTag({ url: `${PYODIDE}pyodide.js` });
  const source = fs.readFileSync(path.join(__dirname, "micro.py"), "utf8");
  const start = Date.now();
  const result = await page.evaluate(async ({ source, indexURL }) => {
    const py = await loadPyodide({ indexURL });
    await py.loadPackage(["numpy", "pandas"]);
    const ns = py.globals.get("dict")();
    ns.set("__name__", "micro");
    py.runPython(source, { globals: ns });
    return JSON.parse(py.runPython("import json; json.dumps(run())", { globals: ns }));
  }, { source, indexURL: PYODIDE });
  result["load and run s"] = (Date.now() - start) / 1000;
  fs.writeFileSync(out, JSON.stringify({ browser: browserName, version: browser.version(), ...result }, null, 1));
  console.log(browserName, browser.version(), JSON.stringify(result));
  await browser.close();
}

async function queries(browserName, site, queriesFile, out, repeats) {
  const server = await serve(site);
  const url = `http://127.0.0.1:${server.address().port}/index.html`;
  const browser = await BROWSERS[browserName].launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e).slice(0, 200)));
  const results = { browser: browserName, version: browser.version(), queries: [], errors };
  const started = Date.now();
  await page.goto(url);
  const states = [];
  let last = "";
  for (;;) {
    const status = await page.textContent("#status");
    const base = status.replace(/ \(\d+ s\)$/, "");
    if (base !== last) { states.push([(Date.now() - started) / 1000, base]); last = base; }
    if (/^Ready|could not start|stopped/.test(status)) break;
    if (Date.now() - started > 1_500_000) break;
    await page.waitForTimeout(500);
  }
  results["startup s"] = (Date.now() - started) / 1000;
  results["startup steps"] = states.map(([t, s]) => `${t.toFixed(0)}s ${s}`);
  console.log(browserName, "ready after", results["startup s"].toFixed(0), "s");
  if (/^Ready/.test(await page.textContent("#status"))) {
    for (const query of JSON.parse(fs.readFileSync(queriesFile, "utf8"))) {
      const runs = [];
      for (let i = 0; i < repeats; i++) {
        await page.fill("#input", query.datalog);
        const t0 = Date.now();
        await page.click("#run");
        let outcome = "ok";
        try {
          await page.waitForFunction(() => !document.getElementById("run").disabled || /stopped|could not/.test(document.getElementById("status").textContent), null, { timeout: 1_200_000, polling: 250 });
        } catch { outcome = "timeout"; }
        const wall = (Date.now() - t0) / 1000;
        const status = await page.textContent("#status");
        const text = await page.textContent("#out");
        const done = /Query completed in ([\d.]+) s/.exec(text);
        if (/stopped|could not/.test(status)) outcome = "worker died: " + status.slice(0, 120);
        else if (/failed/.test(status)) outcome = "failed: " + text.trim().split("\n").slice(-3).join(" ").slice(0, 160);
        runs.push({ s: done ? Number(done[1]) : null, "wall s": wall, outcome });
        console.log(browserName, query.name, JSON.stringify(runs.at(-1)));
        if (outcome.startsWith("worker died") || outcome === "timeout") break;
      }
      results.queries.push({ name: query.name, runs });
      if (runs.at(-1).outcome.startsWith("worker died") || runs.at(-1).outcome === "timeout") break;
    }
  }
  fs.writeFileSync(out, JSON.stringify(results, null, 1));
  await browser.close();
  server.close();
}

const [mode, browserName, ...rest] = process.argv.slice(2);
(mode === "micro" ? micro(browserName, rest[0]) : queries(browserName, rest[0], rest[1], rest[2], Number(rest[3] ?? 1)))
  .catch((error) => { console.error(error); process.exit(1); });
