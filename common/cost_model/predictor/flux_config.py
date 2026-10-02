"""Which CUTLASS config (tile, stages, stream-K mode, raster) Flux uses for a GEMM shape.

Flux takes a tuned config only when (m, n, k) matches a registry entry exactly
(include/flux/op_registry.h:190-203; TuningConfigRegistry::add uses emplace, so the FIRST entry in
the file wins); otherwise it falls back to the first registered hparams (check_heuristic_rule is
always true for GemmV2, src/cuda/op_registry.cu:93-102). The fallback is read from the generated
register file (_GemmHParamsT_0) and also hard-coded below in case the build dir is absent.
Keys: AGKernel (M, n_per_rank, K); ReduceScatter (M, N, k_per_rank).
"""
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

REGISTRY_FILES = {
    "ag": "src/ag_gemm/tuning_config/config_ag_gemm_kernel_sm80_A100_tp{tp}_nnodes1.cu",
    "rs": "src/gemm_rs/tuning_config/config_gemm_rs_sm80_A100_tp{tp}_nnodes1.cu",
}
FALLBACK_FILES = {
    "ag": "build/src/ag_gemm/registers/flux_bf16_bf16_void_bf16_fp32_fp32_sm80_a100_agkernel_rcr_gemmv2_false_nil.cu",
    "rs": "build/src/gemm_rs/registers/flux_bf16_bf16_void_bf16_fp32_fp32_sm80_a100_reducescatter_rcr_gemmv2_false_false_intranode.cu",
}
# parsed 2026-10-02 from FALLBACK_FILES (_GemmHParamsT_0)
FALLBACK_DEFAULT = {
    "ag": {"tile": (128, 128, 64), "stages": 3, "sk": "SK", "raster": "M", "source": "fallback",
           "tuned_for": "none"},
    "rs": {"tile": (128, 128, 32), "stages": 3, "sk": "SK", "raster": "H", "source": "fallback",
           "tuned_for": "none"},
}
_BF16_RCR = "_BF16{}(),_BF16{}(),_Void{}(),_BF16{}()"


def _parse_hparams(s):
    tiles = re.findall(r"make_tuple\((?:cute::Int<)?(\d+)l?(?:>\{\})?,(?:cute::Int<)?(\d+)l?(?:>\{\})?,"
                       r"(?:cute::Int<)?(\d+)l?(?:>\{\})?\)", s)
    g = re.search(r"_GemmStreamK\{\}(?:\(\))?,(?:cute::Int<)?(\d+)(?:>\{\})?,_Raster(\w+?)\{\}", s)
    raster = {"AlongN": "N", "AlongM": "M"}.get(g.group(2), "H") if g else "H"
    return {"tile": tuple(int(x) for x in tiles[-1]), "stages": int(g.group(1)) if g else 3,
            "sk": "DP" if "_StreamkDP" in s else "SK", "raster": raster}


class FluxConfigs:
    def __init__(self, tp=8, repo=REPO):
        self.tp = tp
        self.registry = {"ag": {}, "rs": {}}
        self.fallback = {k: dict(v) for k, v in FALLBACK_DEFAULT.items()}
        self.sources = {}
        for op, rel in REGISTRY_FILES.items():
            path = os.path.join(repo, rel.format(tp=tp))
            self.sources[op] = rel.format(tp=tp)
            if not os.path.exists(path):
                continue
            section = ""
            for line in open(path):
                s = line.strip()
                if s.startswith("//") and "inst.add" not in s:
                    section = s.strip("/ ").strip()  # e.g. "PCIE", "NVLink", "Train", "Inference"
                    continue
                if "inst.add" not in s or s.startswith("//") or _BF16_RCR not in s or "_RCR{}" not in s:
                    continue
                rc = re.search(r"make_runtime_config\((\d+),(\d+),(\d+)", s)
                key = tuple(int(x) for x in rc.groups())
                if key not in self.registry[op]:  # emplace: first entry wins
                    cfg = _parse_hparams(s)
                    cfg["source"] = "registry"
                    cfg["section"] = section
                    # entries under the "PCIE" heading were tuned for a PCIe-only topology
                    cfg["tuned_for"] = "pcie" if "PCIE" in section.upper() else "nvlink/any"
                    self.registry[op][key] = cfg
        for op, rel in FALLBACK_FILES.items():
            path = os.path.join(repo, rel)
            if os.path.exists(path):
                for line in open(path):
                    if "_GemmHParamsT_0 =" in line:
                        cfg = _parse_hparams(line)
                        cfg["source"] = "fallback"
                        cfg["tuned_for"] = "none"
                        self.fallback[op] = cfg
                        break

    def get(self, op, m, n, k):
        return self.registry[op].get((m, n, k), self.fallback[op])

    def is_registry(self, op, m, n, k):
        return (m, n, k) in self.registry[op]
