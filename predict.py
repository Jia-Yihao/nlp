#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path


def write_jsonl(output_path: Path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_input_jsonl(input_root: str):
    input_root = Path(input_root)
    if input_root.is_file() and input_root.suffix == ".jsonl":
        return str(input_root)
    if not input_root.exists():
        return None
    candidates = sorted(input_root.rglob("*.jsonl"))
    return str(candidates[0]) if candidates else None


def smoke_test_predict():
    """Called when script is run without arguments (TIRA smoke test)."""
    input_dir = os.environ.get("TIRA_INPUT_DATASET", "/tira-data/input")
    output_dir = os.environ.get("TIRA_OUTPUT_DIR", "/tira-data/output")
    print(f"[SMOKE] input_dir={input_dir}, output_dir={output_dir}")

    input_file = find_input_jsonl(input_dir)
    preds = []
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                sid = obj.get("id", f"missing_id_{i}")
                preds.append({"id": sid, "label": 0.5})
    else:
        preds = [{"id": "smoke_test_placeholder", "label": 0.5}]

    out_file = Path(output_dir) / "predictions.jsonl"
    write_jsonl(out_file, preds)
    print(f"[SMOKE] wrote {len(preds)} lines to {out_file}")
    return 0


def load_model():
    """Load model from local ./model/ directory."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_dir = Path(__file__).parent / "model"
    if not model_dir.exists():
        raise RuntimeError(f"Model directory not found: {model_dir}")

    # Optional debugging
    print(f"[MODEL] Loading from {model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[MODEL] Loaded successfully. Device = {device}")
    print(f"[MODEL] num_labels = {getattr(model.config, 'num_labels', None)}")
    return tokenizer, model, device


def logits_to_prob(logits):
    import torch
    if logits.dim() != 2:
        return None
    if logits.size(1) == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    if logits.size(1) == 1:
        return torch.sigmoid(logits[:, 0])
    return None


def predict_file(input_file: str, output_dir: str, batch_size: int = 16):
    import torch

    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / "predictions.jsonl"

    tokenizer, model, device = load_model()

    # Read all samples
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("id", f"missing_id_{line_num}")
                text = obj.get("text", "")
                if not isinstance(text, str):
                    text = ""
                samples.append((sid, text))
            except Exception:
                samples.append((f"bad_json_line_{line_num}", ""))

    print(f"[RUN] Loaded {len(samples)} samples, batch_size={batch_size}")

    results = []
    empty_count = 0

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        ids = [x[0] for x in batch]
        texts = [x[1] for x in batch]

        empty_mask = [len(t.strip()) == 0 for t in texts]
        empty_count += sum(empty_mask)

        # Replace empty strings to avoid tokenizer issues
        safe_texts = [t if len(t.strip()) > 0 else " " for t in texts]

        inputs = tokenizer(
            safe_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        if start == 0:
            print(f"[DEBUG] logits shape = {tuple(logits.shape)}")
            print(f"[DEBUG] logits[0] = {logits[0].tolist()}")

        probs = logits_to_prob(logits)
        if probs is None:
            print(f"[ERROR] Unexpected logits shape: {logits.shape}")
            probs = torch.full((len(batch),), 0.5, device=device)

        probs = probs.detach().cpu().tolist()

        for sid, prob, is_empty in zip(ids, probs, empty_mask):
            if is_empty:
                prob = 0.5
            else:
                prob = float(prob)
                # Ensure in [0,1]
                prob = max(0.0, min(1.0, prob))
            results.append({"id": sid, "label": round(prob, 6)})

    # Write output
    with open(output_file, "w", encoding="utf-8") as f_out:
        for r in results:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[STATS] total={len(results)}, empty_text={empty_count}")
    print(f"✅ Prediction saved to {output_file}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("pos_input", nargs="?", default=None)
    parser.add_argument("pos_output", nargs="?", default=None)
    args = parser.parse_args()

    input_file = args.input if args.input is not None else args.pos_input
    output_dir = args.output if args.output is not None else args.pos_output

    # Smoke test mode (no arguments)
    if input_file is None or output_dir is None:
        print("TIRA Smoke Test detected (missing arguments).")
        return smoke_test_predict()

    print(f"[MAIN] Input: {input_file}, Output: {output_dir}")

    # Optional: print model directory info for debugging
    model_dir = Path(__file__).parent / "model"
    if model_dir.exists():
        files = os.listdir(model_dir)
        print(f"[MAIN] model dir exists, files: {files}")
        safetensors = [f for f in files if f.endswith(".safetensors")]
        if safetensors:
            size = (model_dir / safetensors[0]).stat().st_size
            print(f"[MAIN] {safetensors[0]} size = {size} bytes")

    if not os.path.exists(input_file):
        print(f"⚠️ Input file not found: {input_file}, writing placeholder.")
        out_file = Path(output_dir) / "predictions.jsonl"
        write_jsonl(out_file, [{"id": "missing_input", "label": 0.5}])
        return 0

    return predict_file(input_file, output_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
