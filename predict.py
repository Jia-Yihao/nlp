import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os

def main():
    # ====================== TIRA 平台自动适配（核心修复）======================
    # 自动获取输入文件 + 输出目录（TIRA 固定路径，不需要命令行参数）
    input_dir = "/input"
    output_dir = "/output"

    # 自动找输入文件（兼容所有格式）
    input_files = [f for f in os.listdir(input_dir) if f.endswith((".jsonl", ".txt"))]
    if not input_files:
        raise Exception("❌ 未找到输入文件！")
    input_file = os.path.join(input_dir, input_files[0])

    # 输出文件固定路径（TIRA 强制要求）
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "predictions.jsonl")
    # ======================================================================

    # 获取 predict.py 所在的绝对目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model")

    # 加载本地模型和分词器
    print("✅ Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    # 如果有 GPU 就用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"✅ Processing {input_file}...")

    # 打开输入文件和输出文件
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            if not line.strip():
                continue

            data = json.loads(line.strip())
            text_id = data["id"]
            text = data["text"]

            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                probabilities = F.softmax(logits, dim=-1)
                ai_prob = probabilities[0][1].item()

                # 0.5 弃权机制
                if 0.40 <= ai_prob <= 0.60:
                    ai_prob = 0.5

            except Exception as e:
                print(f"⚠️ Error processing {text_id}: {e}")
                ai_prob = 0.5

            result = {"id": text_id, "label": round(ai_prob, 4)}
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"✅ Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
