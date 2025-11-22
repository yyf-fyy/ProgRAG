#!/usr/bin/env python3
"""
测试 Gemma-2-9b-it 模型是否能正常运行
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def test_gemma_model(model_path, device=None):
    """
    测试 Gemma 模型
    
    Args:
        model_path: 模型路径
        device: 指定设备，如 'cuda:2' 或 'cuda:0'，默认为自动选择
    """
    print("=" * 60)
    print("测试 Gemma-2-9b-it 模型")
    print("=" * 60)
    
    model_path = Path(model_path)
    
    # 1. 检查模型路径
    print(f"\n1. 检查模型路径...")
    if not model_path.exists():
        print(f"   ❌ 模型路径不存在: {model_path}")
        return False
    
    print(f"   ✅ 模型路径存在: {model_path}")
    
    # 检查关键文件
    required_files = ['config.json', 'tokenizer_config.json']
    for file in required_files:
        if (model_path / file).exists():
            print(f"   ✅ 找到 {file}")
        else:
            print(f"   ⚠️  未找到 {file}（可能不是必需的）")
    
    # 2. 检查设备
    print(f"\n2. 检查设备...")
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
            print(f"   ✅ 自动选择设备: {device}")
        else:
            device = 'cpu'
            print(f"   ⚠️  CUDA 不可用，将使用 CPU（会很慢）")
    else:
        print(f"   ✅ 使用指定设备: {device}")
    
    if 'cuda' in device:
        device_id = int(device.split(':')[1]) if ':' in device else 0
        if torch.cuda.is_available():
            print(f"   ✅ CUDA 设备: {torch.cuda.get_device_name(device_id)}")
            print(f"   ✅ CUDA 版本: {torch.version.cuda}")
            print(f"   ✅ 可用显存: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.2f} GB")
        else:
            print(f"   ❌ 指定的 CUDA 设备不可用")
            return False
    
    # 3. 加载模型
    print(f"\n3. 加载模型...")
    try:
        print(f"   正在加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        print(f"   ✅ Tokenizer 加载成功")
        
        print(f"   正在加载模型（这可能需要几分钟）...")
        if 'cuda' in device:
            # 如果指定了具体的 CUDA 设备，使用 device_map
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True
            )
            model = model.to(device)
        
        print(f"   ✅ 模型加载成功")
        print(f"   模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试推理
    print(f"\n4. 测试推理...")
    try:
        test_prompt = "What is the capital of France?"
        print(f"   测试问题: {test_prompt}")
        
        # 编码输入
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        
        # 生成回答
        print(f"   正在生成回答...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   ✅ 生成成功")
        print(f"\n   完整输出:")
        print(f"   {response}")
        print(f"\n   回答部分:")
        answer = response[len(test_prompt):].strip()
        print(f"   {answer}")
        
    except Exception as e:
        print(f"   ❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 测试中文（如果支持）
    print(f"\n5. 测试中文支持...")
    try:
        test_prompt_cn = "法国的首都是哪里？"
        print(f"   测试问题: {test_prompt_cn}")
        
        inputs = tokenizer(test_prompt_cn, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(test_prompt_cn):].strip()
        print(f"   ✅ 中文测试成功")
        print(f"   回答: {answer}")
        
    except Exception as e:
        print(f"   ⚠️  中文测试失败（可能不支持）: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 模型测试通过！Gemma-2-9b-it 可以正常使用。")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else '/home/yyf/project/ReaRAG/models/gemma-2-9b-it/'
    device = sys.argv[2] if len(sys.argv) > 2 else None
    test_gemma_model(model_path, device)

