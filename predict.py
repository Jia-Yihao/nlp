import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os

def main():
    # ====================== 终极兼容：TIRA + 本地 双模式 ======================
    try:
        # 优先使用命令行参数（TIRA 测试/运行 都用这个！）
        input_file = sys.argv[1]
        output_dir = sys.argv[2]
        print(f"✅ 使用命令行参数：输入={input_file}, 输出={output_dir}")
    except:
        # 兜底：如果没有参数，尝试 TIRA 固定路径（兼容所有情况）
        input_file = "/input/data.jsonl"
        output_dir = "/output"
        print(f"⚠️ 使用兜底路径：输入={input_file}, 输出={output_dir}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "predictions.jsonl")
    # ======================================================================

    # 获取模型路径（完全不变）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model")

    # 加载模型（不变）
    print("✅ Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 预测逻辑（你的代码完全不变）
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
            
    print(f"✅ 预测完成！结果保存在: {output_file}")

if __name__ == "__main__":
    main()
