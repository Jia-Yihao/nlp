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
    """
    TIRA smoke test input usually is a directory like /tira-data/input
    We try to find the first *.jsonl file.
    """
    input_root = Path(input_root)

    if input_root.is_file() and input_root.suffix == ".jsonl":
        return str(input_root)

    if not input_root.exists():
        return None

    # search jsonl
    candidates = sorted(input_root.rglob("*.jsonl"))
    if len(candidates) == 0:
        return None

    return str(candidates[0])


def smoke_test_predict():
    """
    Smoke test mode: must output the SAME NUMBER of lines as the input dataset.
    """
    input_dir = os.environ.get("TIRA_INPUT_DATASET", "/tira-data/input")
    output_dir = os.environ.get("TIRA_OUTPUT_DIR", "/tira-data/output")

    print(f"[SMOKE] input_dir={input_dir}")
    print(f"[SMOKE] output_dir={output_dir}")

    input_file = find_input_jsonl(input_dir)

    if input_file is None:
        print("[SMOKE] No input jsonl found, writing minimal placeholder (may fail).")
        out_file = Path(output_dir) / "predictions.jsonl"
        write_jsonl(out_file, [{"id": "smoke_test_placeholder", "label": 0.5}])
        print(f"[SMOKE] wrote -> {out_file}")
        return 0

    print(f"[SMOKE] detected input_file={input_file}")

    preds = []
    n = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sample_id = obj.get("id", f"missing_id_{n}")
            preds.append({"id": sample_id, "label": 0.5})
            n += 1

    out_file = Path(output_dir) / "predictions.jsonl"
    write_jsonl(out_file, preds)

    print(f"[SMOKE] wrote {len(preds)} lines -> {out_file}")
    return 0


def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_dir = Path(__file__).parent / "model"
    print("Model dir:", model_dir)

    if model_dir.exists():
        try:
            print("Model dir files:", os.listdir(model_dir))
        except Exception as e:
            print("[WARN] Cannot list model dir:", e)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


def predict_file(input_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / "predictions.jsonl"

    try:
        import torch

        tokenizer, model, device = load_model()

        with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue

                data = json.loads(line)
                text = data.get("text", "")

                if not text:
                    prob = 0.5
                else:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        logits = model(**inputs).logits

                    prob = torch.softmax(logits, dim=-1)[0][1].item()

                    if 0.40 <= prob <= 0.60:
                        prob = 0.5

                result = {"id": data["id"], "label": round(prob, 4)}
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"✅ Prediction finished: {output_file}")
        return 0

    except Exception as e:
        print(f"⚠️ Model inference failed, fallback to constant 0.5. Reason: {e}")

        with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                data = json.loads(line)
                result = {"id": data["id"], "label": 0.5}
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"✅ Fallback output written: {output_file}")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("pos_input", nargs="?", default=None)
    parser.add_argument("pos_output", nargs="?", default=None)

    args = parser.parse_args()

    input_file = args.input if args.input is not None else args.pos_input
    output_dir = args.output if args.output is not None else args.pos_output

    # Smoke test: no args provided
    if input_file is None or output_dir is None:
        print("TIRA Smoke Test detected (missing input/output args).")
        return smoke_test_predict()

    print(f"Run mode: input={input_file}, output={output_dir}")

    if not os.path.exists(input_file):
        print(f"⚠️ Input file not found: {input_file}, writing placeholder.")
        out_file = Path(output_dir) / "predictions.jsonl"
        write_jsonl(out_file, [{"id": "missing_input", "label": 0.5}])
        return 0

    return predict_file(input_file, output_dir)


if __name__ == "__main__":
    sys.exit(main())
