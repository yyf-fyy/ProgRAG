#!/usr/bin/env python3
"""
检查 CWQ 数据集图谱预处理是否完成
"""
from pathlib import Path

def check_preprocess_complete():
    graphs_dir = Path('data/graphs')
    
    required_files = {
        'total_graph_cwq.jsonl': '总图谱文件（所有三元组）',
        'cwq_topic_graph.pickle': '主题实体子图映射',
        'cwq_triple2id.pickle': '三元组到ID的映射',
    }
    
    print("=" * 60)
    print("检查 CWQ 数据集图谱预处理完成情况")
    print("=" * 60)
    
    all_complete = True
    for filename, description in required_files.items():
        filepath = graphs_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename}")
            print(f"   描述: {description}")
            print(f"   大小: {size_mb:.2f} MB")
        else:
            print(f"❌ {filename} - 文件不存在")
            print(f"   描述: {description}")
            all_complete = False
        print()
    
    print("=" * 60)
    if all_complete:
        print("✅ 所有必需文件都已生成，预处理完成！")
        print("\n下一步可以：")
        print("  1. 运行 GNN/get_emb.py 生成 GNN 嵌入")
        print("  2. 训练或使用预训练模型进行推理")
    else:
        print("⚠️  还有文件缺失，请检查预处理流程")
    print("=" * 60)
    
    return all_complete

if __name__ == '__main__':
    check_preprocess_complete()

