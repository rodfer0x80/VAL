from __future__ import annotations
from typing import Final
import logging
import sys

from libval.llm import LLM


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


LOGGER: Final = logging.getLogger("libval")


def main() -> int:
    llm = LLM(logger=LOGGER)

    return 0


if __name__ == "__main__":
    sys.exit(main())
