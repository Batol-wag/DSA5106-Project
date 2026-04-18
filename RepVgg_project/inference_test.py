import copy
import torch
import time
import json
import pandas as pd
from pathlib import Path
from thop import profile
import os
import tempfile

from models.repvgg_net import build_repvgg
from models.baselines import create_resnet18, create_resnet34
from repvgg_with_branches import create_repvgg_with_branches

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# ----------------------------
# DEVICE
# ----------------------------
DEVICE_CPU = torch.device("cpu")
DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPEATS = 50
INPUT_SIZE = (1, 3, 32, 32)

# ----------------------------
# TARGET KEYS
# ----------------------------
MODEL_KEYS = ["A0", "A1", "A2", "B0", "B1", "resnet18", "resnet34"]
STUDENT_KEYS = ["a1_default", "a1_light", "a1_strong", "b1_default", "b1_light", "b1_strong"]

TEACHER_CKPT = "./teacher_resnet18.pth"

def build_teacher():
    model = create_resnet18(10)
    return model

def load_teacher(model, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

# ----------------------------
# FIND CHECKPOINT
# ----------------------------
def find_ckpt(key):
    matches = list(RESULTS_DIR.glob(f"*{key}*.pth"))
    return matches[0] if matches else None


def find_json(key):
    matches = list(RESULTS_DIR.glob(f"*{key}*.json"))
    return matches[0] if matches else None


# ----------------------------
# MODEL BUILDER
# ----------------------------
def build_model(key):
    if key in ["A0", "A1", "A2", "B0", "B1"]:
        json_path = find_json(key)

        a_mult = 1.0
        b_mult = 2.5
        variant = "a"

        if json_path:
            cfg = json.load(open(json_path, "r"))
            variant = cfg.get("repvgg_variant", "a")
            a_mult = cfg.get("a_multiplier", 1.0)
            b_mult = cfg.get("b_multiplier", 2.5)

        return build_repvgg(
            variant=variant,
            num_classes=10,
            deploy=False,
            a_multiplier=a_mult,
            b_multiplier=b_mult,
        )

    if key == "resnet18":
        m = create_resnet18(10)
        return m

    if key == "resnet34":
        m = create_resnet34(10)
        return m

    raise ValueError(key)


# ----------------------------
# LOAD WEIGHTS
# ----------------------------
def load_weights(model, path: str):
    print(f"[LOAD] {path}")

    ckpt = torch.load(path, map_location="cpu")

    # extract ONLY weights if others included
    # state_dict = ckpt["model_state_dict"]

    # load safely
    model.load_state_dict(ckpt, strict=False)
    return model

# ----------------------------
# METRICS
# ----------------------------
def params(model):
    return sum(p.numel() for p in model.parameters())


def flops(model):
    device = next(model.parameters()).device
    x = torch.randn(INPUT_SIZE).to(device)
    f, _ = profile(model, inputs=(x,), verbose=False)
    return f

def latency(model, device):
    model = model.to(device)
    model.eval()

    x = torch.randn(INPUT_SIZE).to(device)

    if device.type == "cuda":
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()

    start = time.time()

    for _ in range(REPEATS):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.time() - start) / REPEATS


def model_size_mb(model):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    torch.save(model.state_dict(), tmp.name)
    size_mb = os.path.getsize(tmp.name) / 1e6
    tmp.close()
    os.remove(tmp.name)
    return size_mb

# ----------------------------
# BENCHMARK
# ----------------------------
def benchmark(name, model, deploy=False):
    print(f"\n===== {name} | deploy={deploy} =====")

    # ALWAYS move model first
    model = model.to(DEVICE_CPU)  # use CPU for FLOPs consistency
    model.eval()

    p = params(model)
    f = flops(model)
    size = model_size_mb(model)

    cpu = latency(model, DEVICE_CPU)

    gpu = None
    if torch.cuda.is_available():
        model_gpu = copy.deepcopy(model).to(DEVICE_GPU)
        gpu = latency(model_gpu, DEVICE_GPU)

    print(f"Params: {p/1e6:.2f} M")
    print(f"Size: {size:.2f} MB")
    print(f"FLOPs: {f/1e9:.2f} G")
    print(f"CPU: {cpu*1000:.2f} ms")
    if gpu:
        print(f"GPU: {gpu*1000:.2f} ms")

    return {
        "model": name,
        "deploy": deploy,
        "params": p,
        "model_size_mb": size,
        "flops": f,
        "cpu_latency": cpu,
        "gpu_latency": gpu,
    }


# ----------------------------
# MAIN
# ----------------------------
def main():
    results = []

    for key in STUDENT_KEYS:
        ckpt = find_ckpt(key)

        if ckpt is None:
            print(f"[WARN] Missing checkpoint for {key}")
            continue

        # print(f"\n[LOAD] {ckpt.name}")

        # for students model
        model = create_repvgg_with_branches()
        # for reproducton model
        # model = build_model(key)
        model = load_weights(model, ckpt)

        # non-deploy (if RepVGG, you need a separate model construction logic)
        results.append(benchmark(key, model, deploy=False))

        # deploy version (only meaningful for RepVGG)
        if key in STUDENT_KEYS:
            model_deploy = repvgg_model_convert(model, do_copy=True)
            results.append(benchmark(key, model_deploy, deploy=True))

    # -----------------------------
    # TEACHER MODEL (ResNet-18)
    # # -----------------------------
    # teacher = build_teacher()
    # teacher = load_teacher(teacher, TEACHER_CKPT)
    #
    # results.append(benchmark("teacher_resnet18", teacher, deploy=False))
    #
    df = pd.DataFrame(results)
    out =  "./results.csv"
    df.to_csv(out, index=False)

    print(f"\n✅ Saved: {out}")


if __name__ == "__main__":
    main()