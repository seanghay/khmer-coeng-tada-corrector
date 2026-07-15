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
function render(text, corrected, sites) {
  const outCps = [...corrected];
  const inCps = [...text];
  // site codepoint index -> {p, changed}
  const siteInfo = new Map(sites.map(([idx, p]) => [idx, { p, changed: inCps[idx] !== outCps[idx] }]));
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
    return;
  }
  const t0 = performance.now();
  const corrected = callString(mod._ctda_correct, text);
  const sites = JSON.parse(callString(mod._ctda_predict_json, text));
  const ms = performance.now() - t0;
  const inCps = [...text];
  const outCps = [...corrected];
  const changed = sites.filter(([idx]) => inCps[idx] !== outCps[idx]).length;
  render(text, corrected, sites);
  stats.textContent = sites.length
    ? `${sites.length} site${sites.length > 1 ? "s" : ""}, ${changed} corrected — ${ms.toFixed(0)} ms`
    : "no ្ត/្ដ sites in this text";
}

let timer = null;
input.addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(update, 150);
});

for (const s of SAMPLES) {
  const b = document.createElement("button");
  b.textContent = s.length > 34 ? s.slice(0, 34) + "…" : s;
  b.addEventListener("click", () => {
    input.value = s;
    update();
  });
  samplesEl.appendChild(b);
}

createCtda().then((m) => {
  if (!m._ctda_init()) {
    output.textContent = "Failed to initialize model.";
    return;
  }
  mod = m;
  input.value = SAMPLES[0];
  update();
});
