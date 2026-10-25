# Contributing

Thanks for helping! OpenMatte is small on purpose: one engine (`openmatte/engine.py`), one CLI, one Nuke node.

## Reporting a problem

Open a GitHub issue with:

- what you ran (the exact `openmatte ...` command, or the Nuke node settings),
- the full console output (in Nuke: the Script Editor lines starting with `[openmatte]`),
- OS, GPU, Nuke version, and `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

Please only attach footage you are allowed to share publicly.

## Sending a change

1. Fork, branch, and keep the change focused on one thing.
2. Run the end-to-end check before opening the PR:

   ```bash
   python tests/test_example.py
   ```

3. Describe what you changed and how you tested it.

Contributions are accepted under the Apache-2.0 license of this repository.

## Model licenses

Do not commit model weights. SAM 2 weights are downloaded from Hugging Face at runtime;
MatAnyone2 stays an optional, separately installed extra because its license is non-commercial.
