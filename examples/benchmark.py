"""Measure load/prepare time and warm end-to-end latency for one chosen input."""

import json
import resource
import time

import numpy as np
from decide import arguments, setup

if __name__ == "__main__":
    parser = arguments()
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if args.runs < 1 or not any((args.text, args.image, args.audio)):
        parser.error("provide an input and a positive --runs value")
    started = time.perf_counter()
    engine, tasks = setup(args)
    ready_ms = (time.perf_counter() - started) * 1000
    with engine:
        engine.decide(tasks, text=args.text, image=args.image, audio=args.audio)
        timings = []
        initial_calls = engine.backend.forward_calls
        for _ in range(args.runs):
            started = time.perf_counter()
            engine.decide(tasks, text=args.text, image=args.image, audio=args.audio)
            timings.append((time.perf_counter() - started) * 1000)
        import torch

        result = {
            "load_and_prepare_ms": ready_ms,
            "warm_runs": args.runs,
            "input": {
                "text_characters": len(args.text or ""),
                "image": bool(args.image),
                "audio": bool(args.audio),
            },
            "latency_p50_ms": float(np.percentile(timings, 50)),
            "latency_p95_ms": float(np.percentile(timings, 95)),
            "forward_calls": engine.backend.forward_calls - initial_calls,
            "process_peak_rss_mib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else None,
            "backend": json.loads(engine.backend.identity),
            "gpu_name": torch.cuda.get_device_name()
            if engine.backend.model.device.type == "cuda"
            else None,
            "note": "Single input; no accuracy evaluation. RSS excludes GPU allocations.",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
