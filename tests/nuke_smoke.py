"""Run the OpenMatte node inside Nuke, headless, on the example clip and check the alpha against ground truth.

Usage (after `python examples/make_example.py`):
    Nuke15.1.exe -t tests/nuke_smoke.py <path-to-.venv-python>
"""
import os
import sys

import nuke

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
sys.path.insert(0, ROOT + "/nuke")
import openmatte_nuke  # noqa: E402

python = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OPENMATTE_PYTHON", "")
frames = ROOT + "/examples/ball/frames/####.png"
truth = ROOT + "/examples/ball/truth"
out = ROOT + "/examples/out_nuke/####.png"

root = nuke.root()
root["first_frame"].setValue(0)
root["last_frame"].setValue(23)
read = nuke.nodes.Read(file=frames, first=0, last=23, origfirst=0, origlast=23)
root["format"].setValue(read.format())
nuke.frame(0)

node = openmatte_nuke.create()
node.setInput(0, read)
node["python"].setValue(python)
h = read.format().height()
node["point"].setValue([120, h - 180])  # Nuke Y is bottom-up; ball centre on frame 0 is (120, 180) top-down
node["click_frame"].setValue(0)
openmatte_nuke.generate(node)

# write the node's RGBA out; tests/test_example.py --nuke scores the alpha channel against ground truth
os.makedirs(os.path.dirname(out), exist_ok=True)
w = nuke.nodes.Write(file=out, file_type="png", channels="rgba", raw=True)
w.setInput(0, node)
nuke.execute(w, 0, 23)
print("NUKE_SMOKE_WROTE", out)
