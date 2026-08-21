# Deploying the AlphaMotion generation Space

1. Create a **private**, Gradio, ZeroGPU Space named `alphamotion-generation`.
2. Create a private model repository named `alphamotion-generation-assets` and
   upload the HMR2 and ViTPose files at the paths listed in `README.md`.
3. Add Space variable `ALPHAMOTION_ASSET_REPO=<owner>/alphamotion-generation-assets`.
4. Create a fine-grained read-only token scoped only to the asset repository
   and add it as Space secret `HF_TOKEN`.
5. Push this directory to the Space repository.
6. Verify the Space API page exposes `/generate_text` and `/generate_video`.
7. Create a read-only token scoped only to the private Space, store it as
   `ALPHAMOTION_GENERATION_TOKEN` on AWS, and set
   `ALPHAMOTION_GENERATION_SPACE=<owner>/alphamotion-generation`.

The original HMR2/ViTPose checkpoints are not committed to Greenwich. They are
kept in the private asset repository because each is about 2.5 GB.
