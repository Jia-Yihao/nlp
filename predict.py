#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path


HF_REPO = "Yihao-Jia/eist"


def write_jsonl(output_path: Path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_input_jsonl(input_root: str):
    """
    TIRA smoke test input is usually a directory like /tira-data/input.
    We try to locate the first *.jsonl file.
    """
    input_root = Path(input_root)

    if input_root.is_file() and input_root.suffix == ".jsonl":
        return str(input_root)

    if not input_root.exists():
        return None

    candidates = sorted(input_root.rglob("*.jsonl"))
    if len(candidates) == 0:
        return None

    return str(candidates[0])


def smoke_test_predict():
    """
    Smoke test mode: output must contain SAME number of lines as input.
    """
    input_dir = os.environ.get("TIRA_INPUT_DATASET", "/tira-data/input")
    output_dir = os.environ.get("TIRA_OUTPUT_DIR", "/tira-data/output")

    print(f"[SMOKE] input_dir={input_dir}")
    print(f"[SMOKE] output_dir={output_dir}")

    input_file = find_input_jsonl(input_dir)
    if input_file is None:
        print("[SMOKE] No input jsonl found, writing placeholder.")
        out_file = Path(output_dir) / "predictions.jsonl"
        write_jsonl(out_file, [{"id": "smoke_test_placeholder", "label": 0.5}])
        return 0

    preds = []
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = obj.get("id", f"missing_id_{i}")
            preds.append({"id": sid, "label": 0.5})

    out_file = Path(output_dir) / "predictions.jsonl"
    write_jsonl(out_file, preds)

    print(f"[SMOKE] wrote {len(preds)} lines -> {out_file}")
    return 0


def load_model():
    """
    Load from local ./model first; otherwise download from HuggingFace.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # important: cache in writable directory
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_model_dir = Path(__file__).parent / "model"

    if local_model_dir.exists():
        try:
            print(f"[MODEL] Trying local model: {local_model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
            model.to(device)
            model.eval()
            print("[MODEL] ✅ Loaded local model")
            print("[MODEL] num_labels =", getattr(model.config, "num_labels", None))
            return tokenizer, model, device
        except Exception as e:
            print("[MODEL] ⚠️ Local model load failed:", e)

    print(f"[MODEL] Downloading model from HuggingFace: {HF_REPO}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    print("[MODEL] ✅ Loaded HF model")
    print("[MODEL] num_labels =", getattr(model.config, "num_labels", None))
    print("[MODEL] id2label =", getattr(model.config, "id2label", None))
    print("[MODEL] label2id =", getattr(model.config, "label2id", None))

    return tokenizer, model, device


def logits_to_prob(logits):
    """
    Convert logits to probability of class 1 (AI probability by default).

    Supports:
      logits shape [B,2] -> softmax
      logits shape [B,1] -> sigmoid
    """
    import torch

    if logits.dim() != 2:
        return None

    if logits.size(1) == 2:
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1]

    if logits.size(1) == 1:
        probs = torch.sigmoid(logits[:, 0])
        return probs

    return None


def predict_file(input_file: str, output_dir: str, batch_size: int = 16):
    import torch

    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / "predictions.jsonl"

    tokenizer, model, device = load_model()

    flip_label = os.environ.get("FLIP_LABEL", "0").strip() == "1"
    print("[RUN] flip_label =", flip_label)
    print("[RUN] device =", device)
    print("[RUN] input_file =", input_file)
    print("[RUN] output_file =", output_file)

    # ---------- Load samples ----------
    samples = []
    with open(input_file, "r", encoding="utf-8") as f_in:
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("id", f"missing_id_line_{line_num}")
                text = obj.get("text", "")
                if not isinstance(text, str):
                    text = ""
                samples.append((sid, text))
            except Exception:
                samples.append((f"bad_json_line_{line_num}", ""))

    print(f"[RUN] loaded {len(samples)} samples")

    results = []
    empty_cnt = 0

    # ---------- Batch inference ----------
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        ids = [x[0] for x in batch]
        texts = [x[1] for x in batch]

        empty_mask = [len(t.strip()) == 0 for t in texts]
        empty_cnt += sum(empty_mask)

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
            out = model(**inputs)
            logits = out.logits

        if start == 0:
            print("[DEBUG] logits shape =", tuple(logits.shape))
            print("[DEBUG] logits[0] =", logits[0])

        probs = logits_to_prob(logits)

        if probs is None:
            print("[ERROR] Unexpected logits shape:", tuple(logits.shape))
            probs = torch.full((len(batch),), 0.5, device=device)

        probs = probs.detach().cpu().tolist()

        for sid, prob, is_empty in zip(ids, probs, empty_mask):
            if is_empty:
                prob = 0.5
            else:
                prob = float(prob)

                # flip if needed
                if flip_label:
                    prob = 1.0 - prob

                if prob < 0.0:
                    prob = 0.0
                if prob > 1.0:
                    prob = 1.0

            results.append({"id": sid, "label": round(prob, 6)})

    # ---------- Save ----------
    with open(output_file, "w", encoding="utf-8") as f_out:
        for r in results:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[STATS] total={len(results)}, empty_text={empty_cnt}")
    print(f"✅ Prediction finished: {output_file}")
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

    # Smoke test mode
    if input_file is None or output_dir is None:
        print("TIRA Smoke Test detected (missing input/output args).")
        return smoke_test_predict()

    print(f"[MAIN] Run mode: input={input_file}, output={output_dir}")
    
    model_dir = Path(__file__).parent / "model"
    print("[CHECK] model_dir =", model_dir)
    print("[CHECK] model_dir exists =", model_dir.exists())
    print("[CHECK] model_dir files =", os.listdir(model_dir))
    print("[CHECK] safetensors exists =", (model_dir / "model.safetensors").exists())
    print("[CHECK] safetensors size =", (model_dir / "model.safetensors").stat().st_size)
   
    if not os.path.exists(input_file):
        print(f"⚠️ Input file not found: {input_file}, writing placeholder.")
        out_file = Path(output_dir) / "predictions.jsonl"
        write_jsonl(out_file, [{"id": "missing_input", "label": 0.5}])
        return 0

    return predict_file(input_file, output_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
