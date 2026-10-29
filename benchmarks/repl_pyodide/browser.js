// Time Pyodide in a browser: the REPL page running the benchmark's queries
// (``queries``) or the CPU benchmark of micro.py (``micro``).
//
//   node browser.js queries BROWSER SITE_DIR QUERIES.json OUT.json [REPEATS]
//   node browser.js micro BROWSER OUT.json
//   node browser.js profile BROWSER SITE_DIR OUT.json   (phases of the engine build)
//
// BROWSER is chromium, firefox or webkit (Playwright's). SITE_DIR is the
// directory of the page with its bundle (see README.md). The page is served
// from a server that lives as long as this script.
const fs = require("fs");
const http = require("http");
const path = require("path");
const { chromium, firefox, webkit } = require("playwright");

const BROWSERS = { chromium, firefox, webkit };
const BROWSER_ENGINES = BROWSERS;
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".zip": "application/zip", ".whl": "application/octet-stream" };

function serve(directory, upload = null) {
  const server = http.createServer((request, response) => {
    if (upload && request.method === "POST" && request.url === "/upload") {
      const file = fs.createWriteStream(upload);
      request.pipe(file);
      return file.on("finish", () => response.writeHead(200).end());
    }
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

// The packages the page's worker loads, as worker.js does.
const PYODIDE_PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "lxml", "requests", "joblib", "pyyaml", "packaging", "micropip"];
const PYPI_PACKAGES = ["lark", "typing_inspect", "rdflib", "nibabel", "rich", "prompt_toolkit", "httpx"];

// Time the phases of building the engine in Pyodide, with profile_build.py.
async function profile(browserName, site, out) {
  const server = await serve(site);
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await BROWSERS[browserName].launch();
  const page = await browser.newPage();
  await page.goto(`${base}/bundle.json`);
  await page.addScriptTag({ url: `${PYODIDE}pyodide.js` });
  const source = fs.readFileSync(path.join(__dirname, "profile_build.py"), "utf8");
  const started = Date.now();
  const steps = [];
  const lines = await page.evaluate(async ({ base, indexURL, source, packages, pypi }) => {
    const times = [];
    const mark = (what) => times.push([performance.now(), what]);
    mark("start");
    const py = await loadPyodide({ indexURL });
    mark("pyodide loaded");
    await py.loadPackage(packages);
    mark("numpy, scipy, pandas, scikit-learn loaded");
    const bundle = await (await fetch(`${base}/bundle.json`)).json();
    const micropip = py.pyimport("micropip");
    for (const name of pypi) await micropip.install(name);
    await micropip.install.callKwargs("nilearn", { deps: false });
    mark("PyPI packages installed");
    for (const wheel of bundle.wheels) {
      py.FS.writeFile(`/tmp/${wheel}`, new Uint8Array(await (await fetch(`${base}/${wheel}`)).arrayBuffer()));
      await micropip.install.callKwargs(`emfs:/tmp/${wheel}`, { deps: false });
    }
    mark("NeuroLang wheels installed");
    const data = new Uint8Array(await (await fetch(`${base}/${bundle.data}`)).arrayBuffer());
    py.FS.mkdirTree("/data");
    py.unpackArchive(data, "zip", { extractDir: "/data" });
    mark("data unpacked");
    const lines = [];
    py.setStdout({ batched: (line) => lines.push(line) });
    py.setStderr({ batched: () => {} });
    py.globals.set("profile_source", source);
    py.runPython(`import sys\nsys.argv = ["profile_build.py", "/data"]\nexec(compile(profile_source, "profile_build.py", "exec"), {"__name__": "__main__"})`);
    mark("engine built");
    lines.push("SETUP " + JSON.stringify(times.map(([t, w], i) => [w, i ? Math.round((t - times[i - 1][0]) / 100) / 10 : 0])));
    return lines;
  }, { base, indexURL: PYODIDE, source, packages: PYODIDE_PACKAGES, pypi: PYPI_PACKAGES });
  const text = lines.filter((l) => !l.startsWith("SETUP")).join("\n");
  const setup = JSON.parse(lines.find((l) => l.startsWith("SETUP")).slice(6));
  fs.writeFileSync(out, JSON.stringify({ browser: browserName, version: browser.version(), "total s": (Date.now() - started) / 1000, setup, build: text }, null, 1));
  console.log(browserName, "total", ((Date.now() - started) / 1000).toFixed(0), "s\nsetup (s per step):", JSON.stringify(setup), "\n" + text);
  await browser.close();
  server.close();
}

// formats.py in Pyodide: parquet, npz and csv. pyarrow is loaded as a Pyodide
// package, which is timed (and its download counted).
async function formats(browserName, out) {
  const browser = await BROWSERS[browserName].launch();
  const page = await browser.newPage();
  await page.goto("about:blank");
  await page.addScriptTag({ url: `${PYODIDE}pyodide.js` });
  const source = fs.readFileSync(path.join(__dirname, "formats.py"), "utf8");
  const result = await page.evaluate(async ({ source, indexURL }) => {
    const py = await loadPyodide({ indexURL });
    await py.loadPackage(["numpy", "pandas"]);
    const before = performance.getEntriesByType("resource").reduce((n, e) => n + (e.transferSize || e.encodedBodySize || 0), 0);
    const t = performance.now();
    let loaded = "pyarrow loaded";
    try { await py.loadPackage("pyarrow"); } catch (error) { loaded = `pyarrow not loadable: ${String(error).slice(0, 120)}`; }
    const loadSeconds = (performance.now() - t) / 1000;
    const after = performance.getEntriesByType("resource").reduce((n, e) => n + (e.transferSize || e.encodedBodySize || 0), 0);
    const ns = py.globals.get("dict")();
    ns.set("__name__", "formats");
    py.runPython(source, { globals: ns });
    const timings = JSON.parse(py.runPython("import json; json.dumps(run())", { globals: ns }));
    return { loaded, "pyarrow load s": Math.round(loadSeconds * 10) / 10, "pyarrow download MB": Math.round((after - before) / 104857.6) / 10, ...timings };
  }, { source, indexURL: PYODIDE });
  fs.writeFileSync(out, JSON.stringify({ browser: browserName, version: browser.version(), ...result }, null, 1));
  console.log(JSON.stringify(result, null, 1));
  await browser.close();
}

// A snapshot of Pyodide's memory with the engine built, and a page started from
// it: what a start would cost if the engine were not built again.
//   node browser.js snapshot BROWSER SITE_DIR OUT.json make|load|heap SNAPSHOT_FILE
// ``heap`` only measures the memory after the build, and how far it compresses.
async function snapshot(browserName, site, out, step, file) {
  const server = await serve(site, step === "make" ? file : null);
  if (step === "load") fs.copyFileSync(file, path.join(site, "snapshot.bin"));
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await BROWSER_ENGINES[browserName].launch();
  const page = await browser.newPage();
  page.on("console", (m) => { if (/snapshot|error/i.test(m.text())) console.log("page:", m.text().slice(0, 200)); });
  await page.goto(`${base}/bundle.json`);
  await page.addScriptTag({ url: `${PYODIDE}pyodide.js` });
  const started = Date.now();
  const result = await page.evaluate(async ({ base, indexURL, step, packages, pypi }) => {
    const times = [];
    const mark = (what) => times.push([what, Math.round(performance.now()) / 1000]);
    mark("start");
    let py;
    if (step === "make" || step === "heap") {
      py = await loadPyodide(step === "make" ? { indexURL, _makeSnapshot: true } : { indexURL });
    } else {
      const response = await fetch(`${base}/snapshot.bin`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      mark(`snapshot downloaded (${(bytes.length / 1048576).toFixed(0)} MB)`);
      py = await loadPyodide({ indexURL, _loadSnapshot: bytes });
      mark("pyodide started from the snapshot");
    }
    if (step === "make" || step === "heap") {
      await py.loadPackage(packages);
      const bundle = await (await fetch(`${base}/bundle.json`)).json();
      const micropip = py.pyimport("micropip");
      for (const name of pypi) await micropip.install(name);
      await micropip.install.callKwargs("nilearn", { deps: false });
      for (const wheel of bundle.wheels) {
        py.FS.writeFile(`/tmp/${wheel}`, new Uint8Array(await (await fetch(`${base}/${wheel}`)).arrayBuffer()));
        await micropip.install.callKwargs(`emfs:/tmp/${wheel}`, { deps: false });
      }
      const data = new Uint8Array(await (await fetch(`${base}/${bundle.data}`)).arrayBuffer());
      py.FS.mkdirTree("/data");
      py.unpackArchive(data, "zip", { extractDir: "/data" });
      mark("environment installed");
      await py.runPythonAsync(`
import json
from pathlib import Path
from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.tui import web
nl = engine_registry.build_engine("neurosynth", Path("/data"), 2.0, yaml_path=web.lightweight_engines_yaml("/tmp"))
repl = web.WebRepl(nl, "neurosynth", 2)
def submit(text):
    return json.dumps({"html": repl.submit(text).html})
submit("ans(s) :- mentions(s, 'pain')")
`);
      mark("engine built");
    }
    const probe = performance.now();
    const answer = py.runPython(`
import os, sys
facts = {"data dir": os.path.exists("/data/neurosynth"), "neurolang_apps imported": "neurolang_apps" in sys.modules}
import json
facts["answer"] = json.loads(submit("ans(s) :- mentions(s, 'pain')"))["html"][-200:]
json.dumps(facts)`);
    mark("first query answered");
    const info = { probeSeconds: Math.round(performance.now() - probe) / 1000, facts: JSON.parse(answer), heapMB: Math.round(py._module.HEAPU8.length / 1048576) };
    if (step === "heap") {
      const heap = py._module.HEAPU8;
      info.heapMB = Math.round(heap.length / 1048576);
      const t = performance.now();
      const gz = await new Response(new Blob([heap]).stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer();
      info.heapGzipMB = Math.round(gz.byteLength / 1048576);
      info.gzipSeconds = Math.round(performance.now() - t) / 1000;
      mark("heap measured");
    }
    if (step === "make") {
      const t = performance.now();
      const snapshotBytes = py.makeMemorySnapshot();
      info.snapshotSeconds = Math.round(performance.now() - t) / 1000;
      info.snapshotMB = Math.round(snapshotBytes.length / 1048576);
      const gz = await new Response(new Blob([snapshotBytes]).stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer();
      info.gzipMB = Math.round(gz.byteLength / 1048576);
      mark("snapshot made");
      await fetch(`${base}/upload`, { method: "POST", body: snapshotBytes });
      mark("snapshot uploaded");
    }
    return { times, ...info };
  }, { base, indexURL: PYODIDE, step, packages: PYODIDE_PACKAGES, pypi: PYPI_PACKAGES });
  result["total s"] = (Date.now() - started) / 1000;
  fs.writeFileSync(out, JSON.stringify({ browser: browserName, step, ...result }, null, 1));
  console.log(JSON.stringify(result, null, 1));
  await browser.close();
  server.close();
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
        const rows = /(\d+) rows?\b/.exec(text);
        runs.push({ s: done ? Number(done[1]) : null, "wall s": wall, rows: rows ? Number(rows[1]) : null, outcome });
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
(mode === "micro" ? micro(browserName, rest[0])
  : mode === "formats" ? formats(browserName, rest[0])
  : mode === "profile" ? profile(browserName, rest[0], rest[1])
    : mode === "snapshot" ? snapshot(browserName, rest[0], rest[1], rest[2], rest[3])
      : queries(browserName, rest[0], rest[1], rest[2], Number(rest[3] ?? 1)))
  .catch((error) => { console.error(error); process.exit(1); });
