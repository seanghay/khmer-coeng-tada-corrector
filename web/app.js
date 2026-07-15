"use strict";

const SAMPLES = [
  "ខ្យល់កន្រ្តាក់គ្របដណ្តប់លើផ្ទៃដី",
  "ក្រសួងបានចេញព្រឹត្តិបត្រស្តីពីស្ថានភាពធាតុអាកាស",
  "សេចក្តីស្រឡាញ់ និងសន្ដិភាពនៅកម្ពុជា",
  "គាត់រស់នៅកណ្តាលទីក្រុងភ្នំពេញ ជាមួយក្រុមគ្រួសារ",
];

const input = document.getElementById("input");
const output = document.getElementById("output");
const stats = document.getElementById("stats");
const samplesEl = document.getElementById("samples");
const cpsIn = document.getElementById("cps-in");
const cpsOut = document.getElementById("cps-out");
const copyBtn = document.getElementById("copy");

let lastCorrected = "";

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // fall through to execCommand
    }
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  ta.remove();
  return ok;
}

copyBtn.addEventListener("click", async () => {
  const ok = await copyText(lastCorrected);
  copyBtn.textContent = ok ? "Copied" : "Copy failed";
  setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
});

const MAX_CPS = 300;
const GLYPH_SUBST = new Map([[0x20, "␣"], [0x0a, "⏎"], [0x09, "⇥"], [0x200b, "ZW"]]);

// One chip per codepoint: glyph on top, U+XXXX below. `siteInfo` marks
// TA/DA sites (changed = black, checked = gray).
function renderCps(el, cps, siteInfo) {
  const frag = document.createDocumentFragment();
  const n = Math.min(cps.length, MAX_CPS);
  for (let i = 0; i < n; ++i) {
    const cp = cps[i].codePointAt(0);
    const chip = document.createElement("span");
    chip.className = "cp";
    const info = siteInfo && siteInfo.get(i);
    if (info) chip.classList.add(info.changed ? "changed" : "checked");
    const ch = document.createElement("span");
    ch.className = "ch";
    ch.textContent = GLYPH_SUBST.get(cp) ?? cps[i];
    const u = document.createElement("span");
    u.className = "u";
    u.textContent = "U+" + cp.toString(16).toUpperCase().padStart(4, "0");
    chip.append(ch, u);
    frag.appendChild(chip);
  }
  if (cps.length > n) {
    const more = document.createElement("span");
    more.className = "more";
    more.textContent = `… +${cps.length - n} more`;
    frag.appendChild(more);
  }
  el.replaceChildren(frag);
}

let mod = null;

function callString(fn, text) {
  const n = mod.lengthBytesUTF8(text) + 1;
  const inPtr = mod._malloc(n);
  mod.stringToUTF8(text, inPtr, n);
  const outPtr = fn(inPtr);
  const result = mod.UTF8ToString(outPtr);
  mod._free(inPtr);
  mod._ctda_free(outPtr);
  return result;
}

// Wrap whole grapheme clusters so Khmer shaping is never broken by the <mark>.
function render(corrected, siteInfo) {
  const seg = new Intl.Segmenter("km", { granularity: "grapheme" });
  const frag = document.createDocumentFragment();
  let cpIndex = 0;
  for (const { segment } of seg.segment(corrected)) {
    const segLen = [...segment].length;
    let info = null;
    for (let k = 0; k < segLen; ++k) {
      const s = siteInfo.get(cpIndex + k);
      if (s && (info === null || s.changed)) info = s;
    }
    if (info) {
      const mark = document.createElement("mark");
      if (info.changed) mark.className = "changed";
      mark.textContent = segment;
      mark.title = `P(ដ) = ${info.p.toFixed(3)}`;
      frag.appendChild(mark);
    } else {
      frag.appendChild(document.createTextNode(segment));
    }
    cpIndex += segLen;
  }
  output.replaceChildren(frag);
}

function update() {
  if (!mod) return;
  const text = input.value;
  if (!text.trim()) {
    output.textContent = "";
    stats.textContent = "";
    cpsIn.textContent = "";
    cpsOut.textContent = "";
    lastCorrected = "";
    copyBtn.disabled = true;
    return;
  }
  const t0 = performance.now();
  const corrected = callString(mod._ctda_correct, text);
  const sites = JSON.parse(callString(mod._ctda_predict_json, text));
  const ms = performance.now() - t0;
  const inCps = [...text];
  const outCps = [...corrected];
  // site codepoint index -> {p, changed}
  const siteInfo = new Map(sites.map(([idx, p]) => [idx, { p, changed: inCps[idx] !== outCps[idx] }]));
  const changed = sites.filter(([idx]) => inCps[idx] !== outCps[idx]).length;
  render(corrected, siteInfo);
  renderCps(cpsIn, inCps, siteInfo);
  renderCps(cpsOut, outCps, siteInfo);
  lastCorrected = corrected;
  copyBtn.disabled = false;
  stats.textContent = sites.length
    ? `${sites.length} site${sites.length > 1 ? "s" : ""}, ${changed} corrected — ${ms.toFixed(0)} ms`
    : "no ្ត/្ដ sites in this text";
}

input.addEventListener("input", update);

for (const s of SAMPLES) {
  const b = document.createElement("button");
  b.textContent = s.length > 34 ? s.slice(0, 34) + "…" : s;
  b.addEventListener("click", () => {
    input.value = s;
    update();
  });
  samplesEl.appendChild(b);
}

createCtda({ locateFile: (path) => path + "?v=5" }).then((m) => {
  if (!m._ctda_init()) {
    output.textContent = "Failed to initialize model.";
    return;
  }
  mod = m;
  input.value = SAMPLES[0];
  update();
});
