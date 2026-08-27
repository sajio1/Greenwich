# Install AlphaMotion from a Git checkout

## Requirements

- Linux or Windows 11 with Python 3.10+ (Python 3.11 recommended)
- Git
- Node.js 20+ and npm for the bundled Data Studio viewer
- NVIDIA CUDA is strongly recommended; CPU mode is supported but slower
- About 3 GB free for the core weights and the sparse robot asset checkout

## One-command Linux setup

```bash
git clone https://github.com/sajio1/Greenwich.git
cd Greenwich
./install.sh
./run.sh
```

Open <http://127.0.0.1:7860>. The installer creates `.venv`, installs the
package, downloads AlphaMotion's release artifacts from Hugging Face, installs
the bundled Body Data Studio worker, and checks out the pinned robot visuals
from GMR. User projects and downloaded data are stored in the operating
system's normal per-user application-data directory, not in the Git checkout.
The core checkpoints, Atlas index, and clean-install fallback library come from
the public [`lloydlei/Greenwich`](https://huggingface.co/lloydlei/Greenwich)
Hugging Face repository; the installer verifies the complete expected file set.

Run `alphamotion doctor` at any time to inspect the installation.

## Windows 11 setup

Open PowerShell in the cloned repository and run:

```powershell
.\install.ps1
.\run.ps1
```

If PowerShell blocks local scripts, run
`Set-ExecutionPolicy -Scope Process Bypass` once in that terminal. Then open
<http://127.0.0.1:7860>.

## Licensed human models

AlphaMotion does not redistribute SMPL-H, SMPL-X, or AMASS. Register and accept
the applicable licenses on the official sites, then install the files you
downloaded:

```bash
.venv/bin/alphamotion install-body-models \
  --smplh-archive ~/Downloads/smplh_300.zip \
  --smplx-model ~/Downloads/SMPLX_NEUTRAL.npz
```

- SMPL-H: <https://mano.is.tue.mpg.de/>
- SMPL-X: <https://smpl-x.is.tue.mpg.de/>
- AMASS: <https://amass.is.tue.mpg.de/>

The installer copies these user-supplied files only into the local
AlphaMotion data directory. It never uploads them.

## Exact-source motion library

Place the user-downloaded archives in one directory with these exact names:

```text
BMLmovi.tar.bz2
CMU.tar.bz2
DanceDB.tar.bz2
KIT.tar.bz2
MOYO_smplh_gendered.zip
```

Build the local exact-source library:

```bash
.venv/bin/alphamotion import-smpl-library --input ~/Downloads/alphamotion-body-data
```

To build only selected sources, repeat `--source`, for example:

```bash
.venv/bin/alphamotion import-smpl-library \
  --input ~/Downloads/alphamotion-body-data \
  --source KIT --source CMU
```

Restart AlphaMotion after the import. It automatically discovers the generated
`imported_smpl` shards. When exact-source shards exist, the legacy curated
fallback is not mixed into the Data Studio library.

## Existing data on another disk

The following optional environment variables override the portable defaults:

```bash
export ALPHAMOTION_DATA=/mnt/large-disk/alphamotion/data
export ALPHAMOTION_CACHE=/mnt/large-disk/alphamotion/cache
export ALPHAMOTION_IMPORTED_LIBRARY="$ALPHAMOTION_DATA/imported_smpl"
export ALPHAMOTION_DATA_STUDIO_ROOT=/mnt/large-disk/body_data
export ALPHAMOTION_DATA_STUDIO_CACHE=/mnt/large-disk/data_studio
```

Never commit these directories, licensed body models, tokens, generated
projects, or SQLite databases to Git.
