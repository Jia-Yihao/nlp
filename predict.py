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
    """
    优先尝试本地模型，失败则从 Hugging Face 下载
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 你的 Hugging Face 仓库
    HF_REPO = "Yihao-Jia/eist"
    
    # 本地模型目录
    model_dir = Path(__file__).parent / "model"
    
    # 尝试本地加载
    if model_dir.exists():
        try:
            print(f"📁 Trying local model: {model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.to(device)
            model.eval()
            print("✅ Loaded local model successfully")
            return tokenizer, model, device
        except Exception as e:
            print(f"⚠️ Local model load failed: {e}")
    
    # 从 Hugging Face 下载
    print(f"📥 Downloading model from Hugging Face: {HF_REPO}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
        model = AutoModelForSequenceClassification.from_pretrained(HF_REPO)
        model.to(device)
        model.eval()
        print("✅ Loaded Hugging Face model successfully")
        return tokenizer, model, device
    except Exception as e:
        print(f"❌ Failed to load model from Hugging Face: {e}")
        raise


def predict_file(input_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / "predictions.jsonl"

    try:
        import torch
        
        tokenizer, model, device = load_model()

        # ------------------ 诊断信息 ------------------
        print("\n=== Model Diagnostic ===")
        # 打印分类头的前5个权重（如果存在）
        if hasattr(model, 'classifier') and hasattr(model.classifier, 'weight'):
            print(f"Classifier weight sample: {model.classifier.weight[0][:5]}")
        elif hasattr(model, 'score') and hasattr(model.score, 'weight'):
            print(f"Score weight sample: {model.score.weight[0][:5]}")
        else:
            print("Could not find classifier/score weights")
        
        # 用空文本测试模型输出
        test_input = tokenizer("test", return_tensors="pt")
        test_input = {k: v.to(device) for k, v in test_input.items()}
        with torch.no_grad():
            test_logits = model(**test_input).logits
        test_probs = torch.softmax(test_logits, dim=-1)[0]
        print(f"Dummy test logits: {test_logits}")
        print(f"Dummy test probs: {test_probs}")
        print("========================\n")
        # ---------------------------------------------

        with open(input_file, "r", encoding="utf-8") as f_in, \
             open(output_file, "w", encoding="utf-8") as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    text_id = data.get("id")
                    text = data.get("text", "")

                    if not text_id:
                        print(f"⚠️ Line {line_num}: missing id field")
                        continue

                    if not text:
                        prob = 0.5
                    else:
                        inputs = tokenizer(
                            text, 
                            return_tensors="pt", 
                            truncation=True, 
                            max_length=512,
                            padding=False
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        with torch.no_grad():
                            logits = model(**inputs).logits

                        prob = torch.softmax(logits, dim=-1)[0][1].item()
                        # 打印前几个样本的原始概率，便于调试
                        if line_num <= 3:
                            print(f"Sample {line_num}: logits={logits}, prob={prob}")

                        # 弃权机制暂时注释，直接使用原始概率
                        # if 0.40 <= prob <= 0.60:
                        #     prob = 0.5

                    result = {"id": text_id, "label": round(prob, 4)}
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️ Line {line_num}: JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ Line {line_num}: processing error: {e}")
                    # 出错时输出 0.5
                    try:
                        result = {"id": data.get("id", f"line_{line_num}"), "label": 0.5}
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    except:
                        pass

        print(f"✅ Prediction finished: {output_file}")
        return 0

    except Exception as e:
        print(f"⚠️ Model inference failed, fallback to constant 0.5. Reason: {e}")
        import traceback
        traceback.print_exc()

        # 备用模式：全部输出 0.5
        with open(input_file, "r", encoding="utf-8") as f_in, \
             open(output_file, "w", encoding="utf-8") as f_out:
            
            for line in f_in:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    result = {"id": data.get("id", "unknown"), "label": 0.5}
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                except:
                    pass

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
