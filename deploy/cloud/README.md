# AlphaMotion small-demo cloud deployment

The economical split deployment keeps the always-on Studio on an AWS CPU
instance and wakes a private Hugging Face ZeroGPU Space only for GENMO work.

```text
Browser --HTTPS--> AWS Studio (AlphaMotion + BodyDataStudio + 150 GB EBS)
                         |
                         +--server-side HF token--> private GENMO ZeroGPU Space
```

The browser sees neither the Hugging Face token nor AWS credentials. Set
`ALPHAMOTION_ACCESS_TOKENS` to a comma-separated list of independent demo
tokens (or use the legacy single `ALPHAMOTION_ACCESS_TOKEN`). AlphaMotion binds
each token to at most three browser-generated P-256 device keys. Device
registrations persist on EBS.

- `aws/`: reproducible Studio image, TLS proxy, persistent-volume layout and
  Ubuntu bootstrap.
- `hf_genmo/`: Gradio/ZeroGPU adapter exposing `/generate_text` and
  `/generate_video` and returning a small, non-pickle NPZ contract.

Start with [the AWS runbook](aws/README.md) after the Space is running.
