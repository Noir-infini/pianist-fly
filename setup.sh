#!/usr/bin/env bash
# pianist-fly — self-contained. The ONLY thing this project needs from the
# internet is the MaleCNS connectome the fly's brain actually is (canonical, not
# some other repo's data). Everything else ships in the repo.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/2  install python deps (numpy, scipy, pandas, pyarrow, numba-less v1) =="
python3 -m pip install --quiet -r requirements.txt

echo "== 2/2  fetch the honest connectome (hash-verified; ~239 MB once) =="
if [ ! -s data/graph.npz ]; then
    python3 scripts/fetch_graph.py
else
    echo "graph already present ($(du -h data/graph.npz | cut -f1)) — skipped"
fi
echo
echo "done. run:  python scripts/build_v1_subgraph.py   then   python src/play_piano.py"
