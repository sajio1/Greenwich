# Deploying the GENMO Space

1. Create a **private**, Gradio, ZeroGPU Space named `alphamotion-genmo`.
2. Create a private model repository named `alphamotion-genmo-assets` and
   upload the HMR2 and ViTPose files at the paths listed in `README.md`.
3. Add Space variable `GENMO_ASSET_REPO=<owner>/alphamotion-genmo-assets`.
4. Create a fine-grained read-only token scoped only to the asset repository
   and add it as Space secret `HF_TOKEN`.
5. Push this directory to the Space repository.
6. Verify the Space API page exposes `/generate_text` and `/generate_video`.
7. Create a read-only token scoped only to the private Space, store it as
   `ALPHAMOTION_GENMO_TOKEN` on AWS, and set
   `ALPHAMOTION_GENMO_SPACE=<owner>/alphamotion-genmo`.

The original HMR2/ViTPose checkpoints are not committed to Greenwich. They are
kept in the private asset repository because each is about 2.5 GB.
