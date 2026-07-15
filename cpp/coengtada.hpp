// coengtada: dependency-free C++17 inference for the Khmer COENG TA/DA BiGRU.
//
// Weights are read zero-copy from a caller-provided buffer (mmap'd file or a
// data segment embedded in the wasm binary) in the format written by
// scripts/export_weights.py. SIMD kernels: NEON (arm64), WASM SIMD128,
// scalar fallback.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ctda {

// ---------------------------------------------------------------- mini-BLAS

#if defined(__ARM_NEON)
#include <arm_neon.h>
inline float dot(const float* a, const float* b, int n) {
  float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  float s = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}
#elif defined(__wasm_simd128__)
#include <wasm_simd128.h>
inline float dot(const float* a, const float* b, int n) {
  v128_t acc0 = wasm_f32x4_splat(0.f), acc1 = wasm_f32x4_splat(0.f);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4)));
  }
  v128_t acc = wasm_f32x4_add(acc0, acc1);
  float s = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
            wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}
#else
inline float dot(const float* a, const float* b, int n) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  float s = s0 + s1 + s2 + s3;
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}
#endif

// y = W x + bias; W row-major (rows x cols).
inline void sgemv(const float* W, const float* x, const float* bias, float* y,
                  int rows, int cols) {
  for (int r = 0; r < rows; ++r) y[r] = dot(W + (size_t)r * cols, x, cols) + bias[r];
}

// out (T x rows) = X (T x cols) @ W^T + bias; the input-projection GEMM.
inline void sgemm_nt(const float* X, const float* W, const float* bias, float* out,
                     int T, int rows, int cols) {
  for (int t = 0; t < T; ++t) sgemv(W, X + (size_t)t * cols, bias, out + (size_t)t * rows, rows, cols);
}

inline float sigmoidf(float x) { return 1.f / (1.f + std::exp(-x)); }

// ------------------------------------------------------------------- model

constexpr int PAD = 0, UNK = 1, LATIN = 2, DIGIT = 3;
constexpr uint32_t COENG = 0x17D2, TA = 0x178F, DA = 0x178A;

struct GruDir {
  const float *w_ih, *w_hh, *b_ih, *b_hh;
};

class Model {
 public:
  int vocab_size = 0, emb_dim = 0, hidden = 0, window = 0;

  // Zero-copy init: `data` must outlive the model.
  bool init(const uint8_t* data, size_t len) {
    if (len < 28 || std::memcmp(data, "CTDA", 4) != 0) return false;
    const uint8_t* p = data + 4;
    uint32_t version = ru32(p);
    vocab_size = (int)ru32(p);
    emb_dim = (int)ru32(p);
    hidden = (int)ru32(p);
    window = (int)ru32(p);
    uint32_t n_entries = ru32(p);
    if (version != 1 || (size_t)(p - data) + n_entries * 8 > len) return false;
    vocab_.reserve(n_entries * 2);
    for (uint32_t i = 0; i < n_entries; ++i) {
      uint32_t cp = ru32(p), id = ru32(p);
      vocab_.emplace(cp, (int)id);
    }
    const int H = hidden, E = emb_dim, I1 = 2 * H;
    const float* w = reinterpret_cast<const float*>(p);
    size_t need = (size_t)vocab_size * E;
    for (int in : {E, I1}) need += 2 * ((size_t)3 * H * in + (size_t)3 * H * H + 6 * H);
    need += (size_t)H * I1 + H + 2 * H + 2;
    if ((size_t)(p - data) + need * 4 > len) return false;
    emb_ = take(w, (size_t)vocab_size * E);
    for (GruDir* g : {&l0f_, &l0b_, &l1f_, &l1b_}) {
      int in = (g == &l0f_ || g == &l0b_) ? E : I1;
      g->w_ih = take(w, (size_t)3 * H * in);
      g->w_hh = take(w, (size_t)3 * H * H);
      g->b_ih = take(w, (size_t)3 * H);
      g->b_hh = take(w, (size_t)3 * H);
    }
    h0_w_ = take(w, (size_t)H * I1);
    h0_b_ = take(w, (size_t)H);
    h1_w_ = take(w, (size_t)2 * H);
    h1_b_ = take(w, 2);
    T_ = 2 * window + 1;
    x0_.resize((size_t)T_ * E);
    proj_f_.resize((size_t)T_ * 3 * H);
    proj_b_.resize((size_t)T_ * 3 * H);
    out0_.resize((size_t)T_ * I1);
    out1_.resize((size_t)T_ * I1);
    h_.resize(H);
    hh_.resize(3 * H);
    head_tmp_.resize(H);
    return true;
  }

  int encode_char(uint32_t cp) const {
    auto it = vocab_.find(cp);
    if (it != vocab_.end()) return it->second;
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return LATIN;
    if ((cp >= '0' && cp <= '9') || (cp >= 0x17E0 && cp <= 0x17E9)) return DIGIT;
    return UNK;
  }

  // P(DA) for the site at the center of `ids` (length 2*window+1).
  float predict_window(const int* ids) {
    const int H = hidden, E = emb_dim, T = T_, I1 = 2 * H;
    for (int t = 0; t < T; ++t)
      std::memcpy(&x0_[(size_t)t * E], emb_ + (size_t)ids[t] * E, E * sizeof(float));
    bigru_layer(l0f_, l0b_, x0_.data(), E, out0_.data());
    bigru_layer(l1f_, l1b_, out0_.data(), I1, out1_.data());
    const float* feat = &out1_[(size_t)window * I1];
    sgemv(h0_w_, feat, h0_b_, head_tmp_.data(), H, I1);
    for (int i = 0; i < H; ++i) head_tmp_[i] = head_tmp_[i] > 0.f ? head_tmp_[i] : 0.f;
    float logits[2];
    sgemv(h1_w_, head_tmp_.data(), h1_b_, logits, 2, H);
    return sigmoidf(logits[1] - logits[0]);
  }

 private:
  static uint32_t ru32(const uint8_t*& p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
  }
  static const float* take(const float*& w, size_t n) {
    const float* r = w;
    w += n;
    return r;
  }

  void gru_step(const GruDir& g, const float* ih, float* h) {
    const int H = hidden;
    sgemv(g.w_hh, h, g.b_hh, hh_.data(), 3 * H, H);
    for (int i = 0; i < H; ++i) {
      float r = sigmoidf(ih[i] + hh_[i]);
      float z = sigmoidf(ih[H + i] + hh_[H + i]);
      float n = std::tanh(ih[2 * H + i] + r * hh_[2 * H + i]);
      h[i] = (1.f - z) * n + z * h[i];
    }
  }

  // x: T x in  ->  out: T x 2H (forward states | backward states)
  void bigru_layer(const GruDir& f, const GruDir& b, const float* x, int in, float* out) {
    const int H = hidden, T = T_;
    sgemm_nt(x, f.w_ih, f.b_ih, proj_f_.data(), T, 3 * H, in);
    sgemm_nt(x, b.w_ih, b.b_ih, proj_b_.data(), T, 3 * H, in);
    std::fill(h_.begin(), h_.end(), 0.f);
    for (int t = 0; t < T; ++t) {
      gru_step(f, &proj_f_[(size_t)t * 3 * H], h_.data());
      std::memcpy(out + (size_t)t * 2 * H, h_.data(), H * sizeof(float));
    }
    std::fill(h_.begin(), h_.end(), 0.f);
    for (int t = T - 1; t >= 0; --t) {
      gru_step(b, &proj_b_[(size_t)t * 3 * H], h_.data());
      std::memcpy(out + (size_t)t * 2 * H + H, h_.data(), H * sizeof(float));
    }
  }

  std::unordered_map<uint32_t, int> vocab_;
  const float *emb_ = nullptr, *h0_w_ = nullptr, *h0_b_ = nullptr, *h1_w_ = nullptr,
              *h1_b_ = nullptr;
  GruDir l0f_{}, l0b_{}, l1f_{}, l1b_{};
  int T_ = 0;
  std::vector<float> x0_, proj_f_, proj_b_, out0_, out1_, h_, hh_, head_tmp_;
};

// ------------------------------------------------------------------- UTF-8

inline std::vector<uint32_t> utf8_decode(const std::string& s) {
  std::vector<uint32_t> cps;
  cps.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    uint8_t c = s[i];
    uint32_t cp;
    int n;
    if (c < 0x80) { cp = c; n = 1; }
    else if ((c >> 5) == 0x6) { cp = c & 0x1F; n = 2; }
    else if ((c >> 4) == 0xE) { cp = c & 0x0F; n = 3; }
    else if ((c >> 3) == 0x1E) { cp = c & 0x07; n = 4; }
    else { cp = 0xFFFD; n = 1; }
    if (i + n > s.size()) { cp = 0xFFFD; n = 1; }
    for (int k = 1; k < n; ++k) {
      uint8_t cc = s[i + k];
      if ((cc >> 6) != 0x2) { cp = 0xFFFD; n = k; break; }
      cp = (cp << 6) | (cc & 0x3F);
    }
    cps.push_back(cp);
    i += n;
  }
  return cps;
}

inline void utf8_append(std::string& out, uint32_t cp) {
  if (cp < 0x80) out += (char)cp;
  else if (cp < 0x800) {
    out += (char)(0xC0 | (cp >> 6));
    out += (char)(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += (char)(0xE0 | (cp >> 12));
    out += (char)(0x80 | ((cp >> 6) & 0x3F));
    out += (char)(0x80 | (cp & 0x3F));
  } else {
    out += (char)(0xF0 | (cp >> 18));
    out += (char)(0x80 | ((cp >> 12) & 0x3F));
    out += (char)(0x80 | ((cp >> 6) & 0x3F));
    out += (char)(0x80 | (cp & 0x3F));
  }
}

// ------------------------------------------------------------------ engine

class Engine {
 public:
  bool load(const uint8_t* data, size_t len) { return model_.init(data, len); }
  const Model& model() const { return model_; }

  // (codepoint index of site consonant, P(DA)) for every COENG TA/DA site.
  std::vector<std::pair<size_t, float>> predict_sites(const std::vector<uint32_t>& cps) {
    std::vector<size_t> sites;
    for (size_t i = 1; i < cps.size(); ++i)
      if (cps[i - 1] == COENG && (cps[i] == TA || cps[i] == DA)) sites.push_back(i);
    std::vector<std::pair<size_t, float>> out;
    if (sites.empty()) return out;
    // Encode the normalized text once (every site consonant becomes TA).
    std::vector<int> ids(cps.size());
    for (size_t i = 0; i < cps.size(); ++i)
      ids[i] = model_.encode_char(i >= 1 && cps[i - 1] == COENG && cps[i] == DA ? TA : cps[i]);
    const int W = model_.window, T = 2 * W + 1;
    std::vector<int> win(T);
    out.reserve(sites.size());
    for (size_t pos : sites) {
      for (int t = 0; t < T; ++t) {
        long j = (long)pos - W + t;
        win[t] = (j < 0 || j >= (long)ids.size()) ? PAD : ids[j];
      }
      out.emplace_back(pos, model_.predict_window(win.data()));
    }
    return out;
  }

  std::string correct(const std::string& utf8) {
    std::vector<uint32_t> cps = utf8_decode(utf8);
    auto preds = predict_sites(cps);
    if (preds.empty()) return utf8;
    for (auto& [pos, p_da] : preds) cps[pos] = p_da >= 0.5f ? DA : TA;
    std::string out;
    out.reserve(utf8.size());
    for (uint32_t cp : cps) utf8_append(out, cp);
    return out;
  }

 private:
  Model model_;
};

}  // namespace ctda
