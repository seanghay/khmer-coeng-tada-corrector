// WASM entry points. The model weights are linked into the wasm binary as a
// data segment (build/model_data.c, generated from artifacts/model.bin), so
// the module is fully self-contained and init is zero-copy.

#include <emscripten.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "coengtada.hpp"

extern "C" {
extern const unsigned char g_model_data[];
extern const unsigned int g_model_len;
}

static ctda::Engine g_engine;

static char* dup_string(const std::string& s) {
  char* r = static_cast<char*>(std::malloc(s.size() + 1));
  std::memcpy(r, s.c_str(), s.size() + 1);
  return r;
}

extern "C" {

EMSCRIPTEN_KEEPALIVE int ctda_init() {
  return g_engine.load(g_model_data, g_model_len) ? 1 : 0;
}

// Returns a malloc'd UTF-8 string; free with ctda_free.
EMSCRIPTEN_KEEPALIVE char* ctda_correct(const char* text) {
  return dup_string(g_engine.correct(text));
}

// JSON [[codepoint_index, p_da], ...]; free with ctda_free.
EMSCRIPTEN_KEEPALIVE char* ctda_predict_json(const char* text) {
  auto preds = g_engine.predict_sites(ctda::utf8_decode(text));
  std::string j = "[";
  char buf[48];
  for (size_t i = 0; i < preds.size(); ++i) {
    std::snprintf(buf, sizeof buf, "%s[%zu,%.4f]", i ? "," : "", preds[i].first,
                  preds[i].second);
    j += buf;
  }
  j += "]";
  return dup_string(j);
}

EMSCRIPTEN_KEEPALIVE void ctda_free(void* p) { std::free(p); }

}  // extern "C"
