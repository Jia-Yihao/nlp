import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
from pathlib import Path

def main():
    # ====================== 解析命令行参数 ======================
    if len(sys.argv) != 3:
        print("错误：需要两个参数")
        print("用法: python predict.py <input_file> <output_dir>")
        sys.exit(1)
    
    input_file = sys.argv[1]   # 输入文件路径（如 /input/dataset.jsonl）
    output_dir = sys.argv[2]    # 输出目录路径（如 /output）
    
    print(f"📂 输入文件: {input_file}")
    print(f"📁 输出目录: {output_dir}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件路径（必须是单个 .jsonl 文件）
    output_file = Path(output_dir) / "predictions.jsonl"
    
    # ====================== 加载模型 ======================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    
    print("✅ Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"📱 Device: {device}")
    
    # ====================== 处理输入文件 ======================
    print(f"🔧 处理: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line.strip())
                text_id = data.get("id")
                text = data.get("text", "")
                
                if not text_id:
                    print(f"⚠️ 行 {line_num}: 缺少 id 字段")
                    continue
                
                if not text:
                    print(f"⚠️ 行 {line_num}: text 为空，使用弃权")
                    ai_prob = 0.5
                else:
                    # 处理输入
                    inputs = tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512,
                        padding=False
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        
                    probabilities = F.softmax(logits, dim=-1)
                    ai_prob = probabilities[0][1].item()
                    
                    # 弃权机制：不确定时回答 0.5
                    if 0.40 <= ai_prob <= 0.60:
                        ai_prob = 0.5
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ 行 {line_num} JSON 解析错误: {e}")
                ai_prob = 0.5
            except Exception as e:
                print(f"⚠️ 行 {line_num} 处理错误: {e}")
                ai_prob = 0.5
            
            # 确保分数在 [0, 1] 范围内
            ai_prob = max(0.0, min(1.0, ai_prob))
            
            result = {"id": text_id, "label": round(ai_prob, 4)}
            f_out.write(json.dumps(result) + '\n')
    
    print(f"✅ 预测完成！结果保存在: {output_file}")

if __name__ == "__main__":
    main()
