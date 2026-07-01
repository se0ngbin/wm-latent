# latent

Workspace for latent-space CBF / world-model research. A single repo that
vendors two projects (with full upstream history preserved via `git subtree`):

- `latent_cbf/` — latent CBF filtering + Dreamer/LE-WM world-model pipeline
  (fork of [CMU-IntentLab/latent_cbf](https://github.com/CMU-IntentLab/latent_cbf)).
- `le-wm/` — LE-WM (JEPA) world model
  (fork of [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)).

Each project keeps its own Python env under `<project>/.venv` (gitignored); recreate with the project's
`uv.lock` / `pyproject.toml`.

Note for Sanjit: maybe u can analyze default dreamer and lewm latent space w/o regularization first, so you don't have to deal with training the world model for now.