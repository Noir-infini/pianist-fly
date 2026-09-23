#!/usr/bin/env python3
"""Make the heavy optional deps (IPython, matplotlib) of flybody lazy imports.

Upstream flybody (TuragaLab) ``flybody/utils.py`` imports IPython and
matplotlib at module load. We install flybody with ``--no-deps`` and never use
its ``display_video``/``rollout_and_render`` helpers, so importing flybody
(e.g. ``TemplateTask`` via src/mirror.py) needn't pull in those heavy packages.

This moves those imports from module scope into ``display_video`` — identical
to the local patch the dev venv uses. Idempotent: no-op if already patched.
"""

from pathlib import Path

EAGER = [
    "from IPython.display import HTML\n",
    "import matplotlib\n",
    "import matplotlib.animation as animation\n",
    "import matplotlib.pyplot as plt\n",
]

LAZY = (
    "    # matplotlib + IPython are heavy optional deps imported lazily here "
    "so that\n"
    "    # importing flybody (e.g. for fruitfly + template_task) never pulls "
    "them in.\n"
    "    from IPython.display import HTML\n"
    "    import matplotlib\n"
    "    import matplotlib.animation as animation\n"
    "    import matplotlib.pyplot as plt\n"
)


def patch_utils(utils_py: Path) -> bool:
    text = utils_py.read_text()
    if "heavy optional deps imported lazily" in text:
        return False
    lines = text.splitlines(keepends=True)
    if not all(any(l == eager for l in lines) for eager in EAGER):
        return False
    kept = [l for l in lines if l not in EAGER]
    text = "".join(kept)
    marker = "def display_video("
    if marker not in text:
        return False
    idx = text.index(marker)
    # Insert after the function's docstring (opening + closing `"""`).
    open_marker = '    """\n'
    open_idx = text.index(open_marker, idx)
    close_marker = '    """\n'
    close_idx = text.index(close_marker, open_idx + len(open_marker))
    insert_at = close_idx + len(close_marker)
    text = text[:insert_at] + "\n" + LAZY + text[insert_at:]
    utils_py.write_text(text)
    return True


def main() -> None:
    import importlib.util
    spec = importlib.util.find_spec("flybody")
    if spec is None or spec.submodule_search_locations is None:
        raise SystemExit("flybody not importable — nothing to patch")
    root = Path(list(spec.submodule_search_locations)[0])
    utils_py = root / "utils.py"
    if patch_utils(utils_py):
        print(f"patched flybody/utils.py: {utils_py}")
    else:
        print(f"flybody/utils.py already lazy (or unchanged): {utils_py}")


if __name__ == "__main__":
    main()