#!/bin/sh
# Build the native CLI (build/ctda) and, if emcc is available, the WASM
# module (web/ctda.js + web/ctda.wasm with the weights embedded).
set -e
cd "$(dirname "$0")/.."
mkdir -p build

echo "== native =="
clang++ -O3 -std=c++17 -Wall -Wextra cpp/main.cpp -o build/ctda
echo "built build/ctda"

command -v emcc >/dev/null || { echo "emcc not found; skipping wasm"; exit 0; }

echo "== wasm =="
python3 - <<'EOF'
data = open("artifacts/model.bin", "rb").read()
with open("build/model_data.cpp", "w") as f:
    f.write('extern "C" const unsigned char g_model_data[] = {')
    f.write(",".join(str(b) for b in data))
    f.write('};\nextern "C" const unsigned int g_model_len = %d;\n' % len(data))
print("generated build/model_data.cpp (%d bytes embedded)" % len(data))
EOF

emcc -O3 -std=c++17 -msimd128 cpp/wasm.cpp build/model_data.cpp \
  -s MODULARIZE=1 -s EXPORT_NAME=createCtda -s ENVIRONMENT=web,worker,node \
  -s ALLOW_MEMORY_GROWTH=1 -s FILESYSTEM=0 \
  -s EXPORTED_RUNTIME_METHODS=UTF8ToString,stringToUTF8,lengthBytesUTF8 \
  -s EXPORTED_FUNCTIONS=_ctda_init,_ctda_correct,_ctda_predict_json,_ctda_free,_malloc,_free \
  -o web/ctda.js
ls -la web/ctda.js web/ctda.wasm
