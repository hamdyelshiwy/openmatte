# Testing the Nuke node interactively

The Nuke node has **not** been tested inside a licensed Nuke session yet. It compiles under Nuke 15.1's
Python 3.10.10, but compiling is not an integration test. These steps are the integration test.
Please report the result (pass, or the exact error) in a GitHub issue.

## 0. Prerequisites

- OpenMatte installed in its venv and `python tests/test_example.py` passing.
- The example clip generated: `python examples/make_example.py` (writes `examples/ball/frames/0000.png` …
  `0023.png`, 640×360, and ground truth in `examples/ball/truth/`).

## 1. Make the node available

Add to `%USERPROFILE%\.nuke\init.py` (create the file if needed), using your own path:

```python
nuke.pluginAddPath("C:/path/to/openmatte/nuke")
```

Optionally, before starting Nuke, set the engine path so you do not have to fill the knob:

```bat
setx OPENMATTE_PYTHON "C:\path\to\openmatte\.venv\Scripts\python.exe"
```

Start Nuke. **Expected:** a **Nodes → OpenMatte → OpenMatte** menu entry.
If it is missing, check the Script Editor for errors from `menu.py`.

## 2. Set up the shot

1. Create a Read node: file `C:/path/to/openmatte/examples/ball/frames/####.png`, frame range 0–23.
2. Project Settings: frame range 0–23, format 640×360 (or "use Read's format").
3. Go to frame 0.

## 3. Create and configure the node

1. Select the Read, then **Nodes → OpenMatte → OpenMatte**. Connect its input to the Read if needed.
2. In the node's **OpenMatte** tab:
   - **engine python:** your venv python (pre-filled if `OPENMATTE_PYTHON` is set).
   - **tracker:** `sam2-small`.
   - **click point:** drag the on-screen point in the Viewer onto the **red ball** at frame 0
     (about x=120, y=180 from the top, i.e. **y≈180 in Nuke's bottom-up coordinates**, since 360−180=180).
   - **click frame:** 0. **frame range:** 0 to 23. **refine:** `none`.
3. Click **Generate matte**.

**Expected:** a progress window, then (first run only) a model download, and lines starting with
`[openmatte]` in the Script Editor ending with `wrote 24 alpha frame(s)`.

## 4. Check the result

1. View the node and press **A** in the Viewer to show the alpha channel.
2. **Expected:** a white disc exactly over the ball on every frame 0–23, black elsewhere.
3. Optional numeric check: add a Write after the node (channels `rgba`, file type png, raw on) to
   `C:/path/to/openmatte/examples/out_nuke/####.png`, render 0–23, then run
   `python tests/test_example.py --nuke`. **Expected:** `Nuke IoU min≥0.90` and `ok`.

## 5. Also try

- **refine: vitmatte** and generate again: edges of the alpha become soft.
- A real plate of your own (keep it private), with 2–3 clicks on the subject.
- A click frame other than the first frame of the range.
- Cancel during a run: the progress window should close and the node should not change.

## Report

Include Nuke version, OS, GPU, which steps passed, and the Script Editor output for any failure.
Known v0.1 behaviour: Nuke's UI is blocked while the engine runs (progress is shown).
