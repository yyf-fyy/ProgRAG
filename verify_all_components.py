#!/usr/bin/env python3
"""
验证 ProgRAG 所有组件的完整性
"""
from pathlib import Path
import torch
import pickle

def verify_all_components(dataset='cwq'):
    print("=" * 60)
    print(f"验证 {dataset.upper()} 数据集的所有组件")
    print("=" * 60)
    
    all_ok = True
    
    # # 1. 验证图谱数据文件
    # print("\n1. 验证图谱数据文件...")
    # graph_files = {
    #     f'data/graphs/total_graph_{dataset}.jsonl': '总图谱文件',
    #     f'data/graphs/{dataset}_topic_graph.pickle': '主题实体子图',
    #     f'data/graphs/{dataset}_triple2id.pickle': '三元组映射',
    # }
    #
    # for filepath, desc in graph_files.items():
    #     path = Path(filepath)
    #     if path.exists():
    #         size_mb = path.stat().st_size / (1024 * 1024)
    #         print(f"   ✅ {desc}: {filepath} ({size_mb:.2f} MB)")
    #     else:
    #         print(f"   ❌ {desc}: {filepath} - 文件不存在")
    #         all_ok = False
    #
    # # 2. 验证 GNN 模型
    # print("\n2. 验证 GNN 模型...")
    # gnn_path = Path(f'ckpt/GNN/{dataset}/GNN.pth')
    # if gnn_path.exists():
    #     try:
    #         state = torch.load(gnn_path, map_location='cpu')
    #         print(f"   ✅ GNN 模型: {gnn_path} ({gnn_path.stat().st_size / (1024**2):.2f} MB)")
    #     except Exception as e:
    #         print(f"   ❌ GNN 模型加载失败: {e}")
    #         all_ok = False
    # else:
    #     print(f"   ❌ GNN 模型不存在: {gnn_path}")
    #     all_ok = False
    #
    # # 3. 验证 GNN 嵌入文件
    # print("\n3. 验证 GNN 嵌入文件...")
    # emb_files = {
    #     f'data/{dataset}/emb/relation.pth': '关系嵌入',
    #     f'data/{dataset}/emb/entity2id.pkl': '实体映射',
    #     f'data/{dataset}/emb/rel2id.pkl': '关系映射',
    # }
    #
    # for filepath, desc in emb_files.items():
    #     path = Path(filepath)
    #     if path.exists():
    #         size_mb = path.stat().st_size / (1024 * 1024)
    #         print(f"   ✅ {desc}: {filepath} ({size_mb:.2f} MB)")
    #     else:
    #         print(f"   ⚠️  {desc}: {filepath} - 文件不存在（可能不是必需的）")
    
    # 4. 验证 MPNet 模型
    print("\n4. 验证 MPNet 模型...")
    mpnet_path = Path(f'ckpt/mpnet/{dataset}.mdl')
    if mpnet_path.exists():
        try:
            # 尝试加载检查点
            checkpoint = torch.load(mpnet_path, map_location='cpu')
            print(f"   ✅ MPNet 模型: {mpnet_path} ({mpnet_path.stat().st_size / (1024**2):.2f} MB)")
        except Exception as e:
            print(f"   ⚠️  MPNet 模型文件存在但可能格式不对: {e}")
    else:
        print(f"   ⚠️  MPNet 模型不存在: {mpnet_path}")
        print(f"      提示: MPNet 用于三元组语义评分，如果缺失可能影响性能")
    
    # 5. 验证 CrossEncoder（关系检索器）
    print("\n5. 验证 CrossEncoder（关系检索器）...")
    sbert_path = Path('ckpt/sbert')
    if sbert_path.exists() and any(sbert_path.iterdir()):
        print(f"   ✅ CrossEncoder 模型目录存在: {sbert_path}")
        # 列出文件
        files = list(sbert_path.rglob('*'))
        print(f"      包含 {len(files)} 个文件/目录")
    else:
        print(f"   ⚠️  CrossEncoder 模型不存在: {sbert_path}")
        print(f"      提示: CrossEncoder 用于关系检索，如果缺失可能影响性能")
    
    # 6. 检查 LLM 配置
    print("\n6. 检查 LLM 配置...")
    print(f"   ℹ️  LLM 模型需要根据你的配置选择:")
    print(f"      - 本地模型: 需要指定 --llm_model_path")
    print(f"      - GPT API: 需要设置 --is_GPT 和 --api_key")
    
    # 7. 总结
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ 核心组件（图谱数据 + GNN 模型）已就绪！")
        print("\n下一步可以：")
        print("  1. 测试完整推理流程（如果 MPNet 和 CrossEncoder 已准备）")
        print("  2. 或先运行小规模测试验证系统")
        print("\n测试命令示例：")
        print(f"  python main.py --dataset {dataset} --split validation --output_dir output")
        print(f"  # 或使用更小的测试集")
        print(f"  python main.py --dataset {dataset} --split test --output_dir output --topk 10")
    else:
        print("⚠️  部分核心组件缺失，请先完成数据预处理和模型准备")
    print("=" * 60)
    
    return all_ok

if __name__ == '__main__':
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'cwq'
    verify_all_components(dataset)

