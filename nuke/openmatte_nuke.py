"""OpenMatte node for Nuke: renders the input, runs the openmatte engine in its own Python, reads the alpha back.

The engine runs out-of-process so torch never loads inside Nuke's interpreter.
"""
import os
import subprocess
import tempfile
import uuid

import nuke

PYTHON_ENV = "OPENMATTE_PYTHON"  # path to the python.exe of the venv where openmatte is installed


def create():
    root = nuke.root()
    g = nuke.nodes.Group(name="OpenMatte1")
    g.begin()
    inp = nuke.nodes.Input(name="Input1")
    read = nuke.nodes.Read(name="AlphaRead")
    read["raw"].setValue(True)  # alpha must not be colour-managed
    copy = nuke.nodes.Copy(name="CopyAlpha", from0="rgba.red", to0="rgba.alpha")
    copy.setInput(0, inp)
    copy.setInput(1, read)
    nuke.nodes.Output().setInput(0, copy)
    g.end()

    g.addKnob(nuke.Tab_Knob("openmatte", "OpenMatte"))
    py = nuke.File_Knob("python", "engine python")
    py.setValue(os.environ.get(PYTHON_ENV, ""))
    py.setTooltip(f"python.exe of the environment where openmatte is installed (or set ${PYTHON_ENV})")
    g.addKnob(py)
    g.addKnob(nuke.Enumeration_Knob("tracker", "tracker", ["sam2-small", "sam2-tiny", "sam2-base", "sam2-large", "sam3"]))
    text = nuke.String_Knob("text", "text prompt")
    text.setTooltip("sam3 only: what to matte, e.g. 'person'")
    g.addKnob(text)
    pt = nuke.XY_Knob("point", "click point")
    pt.setTooltip("sam2: drag onto the object you want to matte, on the 'click frame'")
    g.addKnob(pt)
    g.addKnob(nuke.Int_Knob("click_frame", "click frame"))
    g.addKnob(nuke.Int_Knob("first", "frame range"))
    last = nuke.Int_Knob("last", "")
    last.clearFlag(nuke.STARTLINE)
    g.addKnob(last)
    refine = nuke.Enumeration_Knob("refine", "refine", ["none", "vitmatte", "matanyone2", "videomama"])
    refine.setTooltip("matanyone2 and videomama are NON-COMMERCIAL licenses")
    g.addKnob(refine)
    btn = nuke.PyScript_Knob("generate", "Generate matte", "import openmatte_nuke; openmatte_nuke.generate(nuke.thisNode())")
    btn.setFlag(nuke.STARTLINE)
    g.addKnob(btn)
    cache = nuke.String_Knob("cache", "cache")
    cache.setFlag(nuke.INVISIBLE)
    g.addKnob(cache)

    g["click_frame"].setValue(nuke.frame())
    g["first"].setValue(int(root["first_frame"].value()))
    g["last"].setValue(int(root["last_frame"].value()))
    fmt = root.format()
    g["point"].setValue([fmt.width() / 2, fmt.height() / 2])
    return g


def generate(g):
    python = g["python"].value() or os.environ.get(PYTHON_ENV, "")
    if not python or not os.path.isfile(python):
        nuke.message(f"Set 'engine python' to the python.exe where openmatte is installed (or ${PYTHON_ENV}).")
        return
    if g.input(0) is None:
        nuke.message("Connect an input first.")
        return
    first, last, click = int(g["first"].value()), int(g["last"].value()), int(g["click_frame"].value())
    if not first <= click <= last:
        nuke.message("click frame must be inside the frame range.")
        return

    cache = g["cache"].value() or os.path.join(tempfile.gettempdir(), "openmatte", uuid.uuid4().hex[:8])
    g["cache"].setValue(cache)
    in_dir, out_dir = os.path.join(cache, "in"), os.path.join(cache, f"alpha_{uuid.uuid4().hex[:6]}")
    os.makedirs(in_dir, exist_ok=True)

    # 1. render the input frames
    with g:
        w = nuke.nodes.Write(file=os.path.join(in_dir, "####.png").replace("\\", "/"), file_type="png")
        w["datatype"].setValue("16 bit")
        w.setInput(0, nuke.toNode("Input1"))
        try:
            nuke.execute(w, first, last)
        finally:
            nuke.delete(w)

    # 2. run the engine (Nuke Y is bottom-up, the engine wants top-down pixels)
    height = g.input(0).format().height()
    x, y = g["point"].value()
    tracker = g["tracker"].value()
    cmd = [python, "-m", "openmatte", in_dir, "-o", out_dir, "--tracker", tracker,
           "--refine", g["refine"].value(), "-f", str(click - first)]
    if tracker == "sam3":
        if not g["text"].value().strip():
            nuke.message("sam3 needs a text prompt.")
            return
        cmd += ["--text", g["text"].value().strip()]
    else:
        cmd += ["-p", f"{x:.1f},{height - y:.1f}"]
    task = nuke.ProgressTask("OpenMatte")
    task.setMessage("running (first run downloads the models)...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log = []
    for line in proc.stdout:
        log.append(line)
        print("[openmatte]", line.rstrip())
        task.setMessage(line.strip()[:120])
        if task.isCancelled():
            proc.kill()
            break
    code = proc.wait()
    del task
    if code != 0:
        nuke.message("OpenMatte failed:\n\n" + "".join(log[-15:]))
        return

    # 3. read the alpha back
    with g:
        read = nuke.toNode("AlphaRead")
        read["file"].setValue(os.path.join(out_dir, "####.png").replace("\\", "/"))
        for k in ("first", "origfirst"):
            read[k].setValue(first)
        for k in ("last", "origlast"):
            read[k].setValue(last)
        read["reload"].execute()
