# Bundled Body Data Studio runtime

This directory contains the runtime used by AlphaMotion's Data Studio view.
`alphamotion setup` copies it into the user's writable AlphaMotion data
directory and installs the pinned three.js package there. Source motion data,
licensed body models, caches, databases, and generated outputs are deliberately
not bundled.

Run Body Data Studio through AlphaMotion rather than launching this directory
directly. Configuration and user-data locations are documented in
`docs/INSTALL.md` at the repository root.
