// Native CLI: mmap the weight file and correct text from argv or stdin.
//
//   ctda [-m model.bin] [-p] [text ...]     (-p prints per-site P(DA) instead)

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <string>

#include "coengtada.hpp"

int main(int argc, char** argv) {
  std::string model_path = "artifacts/model.bin";
  bool probs = false;
  std::vector<std::string> texts;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-m" && i + 1 < argc) model_path = argv[++i];
    else if (a == "-p") probs = true;
    else texts.push_back(a);
  }

  int fd = open(model_path.c_str(), O_RDONLY);
  if (fd < 0) {
    std::fprintf(stderr, "cannot open %s\n", model_path.c_str());
    return 1;
  }
  struct stat st;
  fstat(fd, &st);
  void* map = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) {
    std::perror("mmap");
    return 1;
  }

  ctda::Engine engine;
  if (!engine.load(static_cast<const uint8_t*>(map), st.st_size)) {
    std::fprintf(stderr, "bad model file %s\n", model_path.c_str());
    return 1;
  }

  auto process = [&](const std::string& line) {
    if (probs) {
      auto cps = ctda::utf8_decode(line);
      for (auto& [pos, p] : engine.predict_sites(cps))
        std::printf("%zu\t%.6f\n", pos, p);
    } else {
      std::fputs(engine.correct(line).c_str(), stdout);
      std::fputc('\n', stdout);
    }
  };

  if (!texts.empty())
    for (auto& t : texts) process(t);
  else
    for (std::string line; std::getline(std::cin, line);) process(line);
  return 0;
}
