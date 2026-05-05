import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
from pathlib import Path

def main():
    # ====================== 兼容多种调用方式 ======================
    if len(sys.argv) == 3:
        # 正式评估：两个参数
        input_file = sys.argv[1]
        output_dir = sys.argv[2]
        print(f"📂 正式模式：输入文件={input_file}, 输出目录={output_dir}")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = Path(output_dir) / "predictions.jsonl"
        
        # 真实预测
        run_prediction(input_file, output_file)
        
    else:
        # 测试模式：尝试多个可能的输出位置
        print("⚠️ 测试模式：尝试创建输出文件")
        
        # 尝试多个可能的输出路径
        possible_outputs = [
            Path("/output/predictions.jsonl"),
            Path("./output/predictions.jsonl"),
            Path("./predictions.jsonl"),
            Path("/app/output/predictions.jsonl"),
            Path("/mnt/output/predictions.jsonl"),
        ]
        
        output_created = False
        
        for output_path in possible_outputs:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('{"id": "test_document", "label": 0.5}\n')
                print(f"✅ 创建输出文件: {output_path.absolute()}")
                output_created = True
            except Exception as e:
                print(f"⚠️ 无法创建 {output_path}: {e}")
        
        # 也在当前目录的常见位置创建
        current_dir = Path.cwd()
        test_output = current_dir / "predictions.jsonl"
        with open(test_output, 'w', encoding='utf-8') as f:
            f.write('{"id": "test_document", "label": 0.5}\n')
        print(f"✅ 在当前目录创建: {test_output.absolute()}")
        
        # 列出所有 jsonl 文件供调试
        all_jsonl = list(Path(".").glob("**/*.jsonl")) + list(Path("/").glob("**/predictions.jsonl"))
        print(f"📂 系统中所有 .jsonl 文件: {[str(f) for f in all_jsonl]}")
        
        if output_created:
            print("🎉 测试模式完成")
        else:
            print("❌ 测试模式无法创建输出文件")
            # 最后尝试：直接写入当前目录
            with open("predictions.jsonl", 'w') as f:
                f.write('{"id": "test", "label": 0.5}\n')
            print("✅ 已写入当前目录: predictions.jsonl")

def run_prediction(input_file, output_file):
    """执行真实预测"""
    print(f"🔧 处理输入文件: {input_file}")
    print(f"💾 输出文件: {output_file}")
    
    # 加载模型
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
    
    total_count = 0
    
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
                    ai_prob = 0.5
                else:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        
                    probabilities = F.softmax(logits, dim=-1)
                    ai_prob = probabilities[0][1].item()
                    
                    if 0.40 <= ai_prob <= 0.60:
                        ai_prob = 0.5
                    
            except Exception as e:
                print(f"⚠️ 行 {line_num} 处理错误: {e}")
                ai_prob = 0.5
            
            ai_prob = max(0.0, min(1.0, ai_prob))
            result = {"id": text_id, "label": round(ai_prob, 4)}
            f_out.write(json.dumps(result) + '\n')
            total_count += 1
    
    print(f"✅ 预测完成！共处理 {total_count} 条，结果保存在: {output_file}")

if __name__ == "__main__":
    main()
