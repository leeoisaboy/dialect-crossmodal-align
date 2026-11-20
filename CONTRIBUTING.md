# Contributing

1. Install with `python -m pip install -e .`.
2. Run `python -m compileall -q .`.
3. Run `python -m pytest -q` when tests are available locally.
4. Keep new alignment methods in dense vector space and expose them through a
   configuration-driven module.
5. Keep retrieval and alignment inside the dense-vector benchmark boundary, and
   do not add licensed data.
