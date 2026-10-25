# Third-party notices

OpenMatte's own source code is licensed under Apache-2.0 (see [LICENSE](LICENSE)).

**This repository does not contain or redistribute any third-party model weights or third-party source code.**
Python packages are installed by `pip` from their publishers, and model weights are downloaded by the user, on
first use, from the locations below. Each is governed by its own license, which the user must accept and follow.
Code licenses and model-weight licenses are listed separately because they can differ.

This file is a summary for convenience, not legal advice. The linked license texts are authoritative.

## Models

| Component | Code license | Weights license | Weights source | Commercial use |
|---|---|---|---|---|
| SAM 2 / SAM 2.1 (Meta) | Apache-2.0 ([LICENSE](https://github.com/facebookresearch/sam2/blob/main/LICENSE)) | Apache-2.0 (same file; [model card](https://huggingface.co/facebook/sam2.1-hiera-small)) | `facebook/sam2.1-hiera-{tiny,small,base-plus,large}` | Yes |
| SAM 3 (Meta) | SAM License ([LICENSE](https://github.com/facebookresearch/sam3/blob/main/LICENSE)) | SAM License; gated access on [Hugging Face](https://huggingface.co/facebook/sam3) | `facebook/sam3` | Allowed by the SAM License, subject to its terms (e.g. trade-control and use restrictions) |
| ViTMatte (HUST VL) | MIT ([repo](https://github.com/hustvl/ViTMatte)) | Apache-2.0 per the [model card](https://huggingface.co/hustvl/vitmatte-small-composition-1k) | `hustvl/vitmatte-small-composition-1k` | Yes |
| MatAnyone2 (NTU S-Lab) | S-Lab License 1.0 ([LICENSE.txt](https://github.com/pq-yang/MatAnyone2/blob/main/LICENSE.txt)) | S-Lab License 1.0 | GitHub release `v1.0.0/matanyone2.pth` | **No.** Redistribution and use permitted "for non-commercial purpose"; commercial use requires contacting the authors |
| VideoMaMa (KAIST CVLab) | CC BY-NC 4.0 ([README, License section](https://github.com/cvlab-kaist/VideoMaMa#-license)) | Stability AI Community License ([text](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/main/LICENSE.md)) | `SammyLim/VideoMaMa` | **No** (the code is NonCommercial) |
| Stable Video Diffusion img2vid-xt (Stability AI), loaded by VideoMaMa | — | Stability AI Community License ([text](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/LICENSE.md)) | `stabilityai/stable-video-diffusion-img2vid-xt` (feature extractor, image encoder, VAE) | Only via VideoMaMa here, which is non-commercial. The license itself requires registration for commercial use and ends above USD 1M annual revenue |

Required attributions when the corresponding optional feature is used:

- **Stable Video Diffusion / VideoMaMa:** "This Stability AI Model is licensed under the Stability AI Community
  License, Copyright © Stability AI Ltd. All Rights Reserved." Powered by Stability AI.
- **MatAnyone2:** Copyright 2023 S-Lab.
- **SAM 3:** use is subject to the SAM License; a copy must accompany any redistribution of SAM Materials
  (OpenMatte does not redistribute them).

SAM 2's optional CUDA extension includes code derived from `cc_torch`
([LICENSE_cctorch](https://github.com/facebookresearch/sam2/blob/main/LICENSE_cctorch)); OpenMatte's install
instructions skip building it.

## Python packages (installed by pip, not redistributed)

From the package metadata of the tested versions (see `requirements-tested.txt`):

| Package | License |
|---|---|
| torch, torchvision | BSD-3-Clause / BSD |
| transformers, diffusers, accelerate, huggingface_hub | Apache-2.0 |
| opencv-python-headless | Apache-2.0 |
| numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| pillow | MIT-CMU |
| einops, hydra-core, iopath | MIT |
| omegaconf | BSD |

## Example data

`examples/make_example.py` generates the example clip procedurally; it contains no third-party images.
`docs/demo_ball.jpg` is rendered from that generated clip and OpenMatte's own output.
