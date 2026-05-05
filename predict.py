import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
from pathlib import Path

def main():
    # ====================== TIRA 标准路径 ======================
    # TIRA 总是将数据挂载在 /input 目录
    # 输出必须写入 /output 目录
    input_path = '/input'
    output_dir = '/output'
    
    print(f"📂 使用 TIRA 标准路径：输入={input_path}, 输出={output_dir}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} 不存在，尝试当前目录")
        input_path = '.'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # ====================== 查找输入文件 ======================
    input_files = []
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        input_files = [input_path_obj]
    elif input_path_obj.is_dir():
        # 查找所有 .jsonl 文件
        input_files = list(input_path_obj.glob("*.jsonl"))
        if not input_files:
            input_files = list(input_path_obj.glob("*.txt"))
    
    if not input_files:
        # 列出目录内容帮助调试
        print(f"❌ 在 {input_path} 中找不到 .jsonl 或 .txt 文件")
        if os.path.exists(input_path):
            print(f"目录内容: {list(Path(input_path).iterdir())}")
        raise FileNotFoundError(f"在 {input_path} 中找不到输入文件")
    
    print(f"📂 找到 {len(input_files)} 个输入文件: {[f.name for f in input_files]}")
    
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

    # ====================== 处理每个输入文件 ======================
    for input_file in input_files:
        # TIRA 期望的输出文件名
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
                        
                except Exception as e:
                    print(f"⚠️ 行 {line_num} 处理错误: {e}")
                    ai_prob = 0.5
                    
                result = {"id": text_id, "label": round(ai_prob, 4)}
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"✅ 完成: {output_file.name}")
    
    print(f"🎉 所有预测完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
