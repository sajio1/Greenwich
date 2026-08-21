# Local deployment

This checkout is pinned to Greenwich commit `f4024a7`.

```bash
./deploy/doctor.sh
./deploy/run.sh
```

The UI is served at <http://127.0.0.1:7860>.

Large environments, caches, models, and robot assets live under
`/media/sajio/New Volume/CodexDeployments/Greenwich`; the source checkout stays
on the system disk. `deploy/env.sh` contains the runtime configuration.

Motion generation runs in a separate Python environment. Both Add a Motion
(text) and Upload Video to Generate use AlphaMotion and return the same global
SMPL-22 rotations and root-trajectory contract consumed by the Greenwich
retargeter.

Configure the generation paths through `ALPHAMOTION_GENERATION_REPO` and
`ALPHAMOTION_GENERATION_PYTHON`. Text generation also requires a short
camera/reference clip in the AlphaMotion cache. AlphaMotion does not bundle or
re-host third-party weights.
