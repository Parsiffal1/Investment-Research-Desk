from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from typing import Any


REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
    "evaluate",
    "sklearn",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the WSL CUDA LoRA training environment.")
    parser.add_argument("--load-model", action="store_true", help="Download and load Qwen/Qwen3-8B with 4-bit NF4 quantization.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Base model to use for the optional 4-bit load test.")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "python": sys.version,
        "missing_packages": [package for package in REQUIRED_PACKAGES if importlib.util.find_spec(package) is None],
    }
    if result["missing_packages"]:
        print(json.dumps(result, indent=2))
        return 1

    import torch
    from transformers import BitsAndBytesConfig

    result["torch_version"] = torch.__version__
    result["cuda_available"] = torch.cuda.is_available()
    result["cuda_device_count"] = torch.cuda.device_count()
    result["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    result["bf16_supported"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    result["bitsandbytes_4bit_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ).to_dict()

    if not result["cuda_available"]:
        print(json.dumps(result, indent=2, default=str))
        return 1

    if args.load_model:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map="auto",
            trust_remote_code=True,
        )
        result["model_load_test"] = {
            "model": args.model,
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
        }

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
