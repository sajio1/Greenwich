# Local deployment

The portable public setup is documented in `docs/INSTALL.md`.

```bash
./deploy/doctor.sh
./deploy/run.sh
```

The UI is served at <http://127.0.0.1:7860>.

By default, environments live in `.venv` and runtime data uses the operating
system's per-user application-data directories. `ALPHAMOTION_DATA` and
`ALPHAMOTION_CACHE` can place large assets on another disk.

Motion generation runs in a separate Python environment. Both Add a Motion
(text) and Upload Video to Generate use AlphaMotion and return the same global
SMPL-22 rotations and root-trajectory contract consumed by the Greenwich
retargeter.

Configure the generation paths through `ALPHAMOTION_GENERATION_REPO` and
`ALPHAMOTION_GENERATION_PYTHON`. Text generation also requires a short
camera/reference clip in the AlphaMotion cache. AlphaMotion does not bundle or
re-host third-party weights.
