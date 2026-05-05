import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
from pathlib import Path

def main():
    # ====================== 处理输入输出路径 ======================
    if len(sys.argv) >= 3:
        # TIRA 标准调用：python predict.py --input /path/to/data --output /path/to/output
        # 或者：python predict.py /path/to/data /path/to/output
        input_path = sys.argv[1]
        output_dir = sys.argv[2]
        
        # 处理 --input 和 --output 参数格式
        if input_path == '--input' and len(sys.argv) >= 4:
            input_path = sys.argv[2]
            output_dir = sys.argv[4] if len(sys.argv) >= 5 else output_dir
        
        print(f"✅ 使用命令行参数：输入={input_path}, 输出={output_dir}")
    else:
        # 兜底：尝试环境变量或默认路径
        input_path = os.environ.get('INPUT_DIR', '/input')
        output_dir = os.environ.get('OUTPUT_DIR', '/output')
        print(f"⚠️ 使用环境变量/默认路径：输入={input_path}, 输出={output_dir}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # ====================== 查找输入文件 ======================
    input_files = []
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        # 如果是文件，直接使用
        input_files = [input_path_obj]
    elif input_path_obj.is_dir():
        # 如果是目录，查找所有 .jsonl 文件
        input_files = list(input_path_obj.glob("*.jsonl"))
        if not input_files:
            input_files = list(input_path_obj.glob("*.txt"))
    
    if not input_files:
        raise FileNotFoundError(f"在 {input_path} 中找不到 .jsonl 或 .txt 文件")
    
    print(f"📂 找到 {len(input_files)} 个输入文件: {[f.name for f in input_files]}")
    
    # ====================== 加载模型 ======================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model")
    
    print("✅ Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"📱 Device: {device}")

    # ====================== 处理每个输入文件 ======================
    for input_file in input_files:
        # 输出文件名：原文件名_predictions.jsonl
        output_file = Path(output_dir) / f"{input_file.stem}_predictions.jsonl"
        
        print(f"🔧 处理: {input_file.name} -> {output_file.name}")
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line.strip())
                    text_id = data.get("id", f"line_{line_num}")
                    text = data.get("text", "")
                    
                    if not text:
                        print(f"⚠️ 行 {line_num}: 缺少 text 字段")
                        ai_prob = 0.5
                    else:
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
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ 行 {line_num} JSON 解析错误: {e}")
                    ai_prob = 0.5
                except Exception as e:
                    print(f"⚠️ 行 {line_num} 处理错误: {e}")
                    ai_prob = 0.5
                    
                result = {"id": text_id, "label": round(ai_prob, 4)}
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            print(f"✅ 完成: {output_file.name}")
    
    print(f"🎉 所有预测完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
