# pianist-fly

MaleCNS fly connectome research: a fly that plays the piano with its own legs.

- Brain: real MaleCNS v1 connectome (239 MB, fetched+verified — never committed).
- Fly sculpture + 5t legs + dm_control/mujoco run in the TRANSPLANTED COPY (
  `src/`, doom-fire-free), drive = NPi-electrode (not doom).
- Drive: scent enters the antenna (olfactory ORNs), brain drives legs natively.
- Honesty: every result is `source.kind: predicted` from our own connectome math —
  no doom, no decoder-as-conductor, no claimed edge. See `docs/transplant.md`.

Run:
    ./setup.sh          # deps + fetch connectome
    python src/play_piano.py
