"""CLI demo: correct COENG TA/DA in text from arguments or stdin.

    uv run python main.py "ខ្យល់គ្របដណ្តប់លើផ្ទៃដី"
    echo "ស្តីពីការងារ" | uv run python main.py
"""

import sys

from coengtada import Corrector


def main():
    corrector = Corrector()
    texts = sys.argv[1:] or (line.rstrip("\n") for line in sys.stdin)
    for text in texts:
        print(corrector.correct(text))


if __name__ == "__main__":
    main()
