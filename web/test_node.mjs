// Node parity test: run the WASM engine over a file and print corrections.
//   node web/test_node.mjs <input.txt>
import { readFileSync } from "node:fs";
import createCtda from "./ctda.js";

const mod = await createCtda();
if (!mod._ctda_init()) throw new Error("ctda_init failed");

const call = (fn, text) => {
  const n = mod.lengthBytesUTF8(text) + 1;
  const inPtr = mod._malloc(n);
  mod.stringToUTF8(text, inPtr, n);
  const outPtr = fn(inPtr);
  const out = mod.UTF8ToString(outPtr);
  mod._free(inPtr);
  mod._ctda_free(outPtr);
  return out;
};

const lines = readFileSync(process.argv[2], "utf-8").split("\n");
const t0 = performance.now();
let sites = 0;
for (const line of lines) {
  console.log(call(mod._ctda_correct, line));
  sites += JSON.parse(call(mod._ctda_predict_json, line)).length;
}
console.error(`${sites} sites in ${((performance.now() - t0) / 1000).toFixed(2)}s`);
