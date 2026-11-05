"""Convert a '# %%'-style script into a Jupyter/Colab notebook.

Usage: python tools/py_to_ipynb.py scripts/02_deep_audit.py notebooks/02_deep_audit.ipynb

- Text before the first '# %%' (the copy-paste instructions) is skipped.
- '# %% [markdown]' blocks become markdown cells (leading '# ' removed).
- '# %%' blocks become code cells.
"""
import json
import re
import sys
from pathlib import Path


def convert(src: Path, dst: Path) -> int:
    text = src.read_text()
    blocks = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    # blocks = [preamble, tag1, body1, tag2, body2, ...]
    for tag, body in zip(blocks[1::2], blocks[2::2]):
        body = body.strip("\n")
        if "[markdown]" in tag:
            lines = []
            for line in body.split("\n"):
                if line.startswith("# "):
                    lines.append(line[2:])
                elif line.strip() == "#":
                    lines.append("")
                else:
                    lines.append(line)
            # re-join soft-wrapped prose lines: keep line breaks (markdown renders them as spaces)
            cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": body})
    nb = {"cells": cells,
          "metadata": {"colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, indent=1))
    return len(cells)


if __name__ == "__main__":
    n = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Wrote {sys.argv[2]} with {n} cells")
