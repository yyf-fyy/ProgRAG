#!/usr/bin/env python3
"""
验证 cwq_topic_graph.pickle 文件是否正确生成
"""
import pickle
from pathlib import Path

def verify_topic_graph():
    # 检查文件
    topic_graph_path = Path('data/graphs/cwq_topic_graph.pickle')
    triple2id_path = Path('data/graphs/cwq_triple2id.pickle')

    print("=" * 60)
    print("验证 cwq_topic_graph.pickle")
    print("=" * 60)

    # 1. 检查文件是否存在
    if not topic_graph_path.exists():
        print(f"❌ 文件不存在: {topic_graph_path}")
        return False
    print(f"✅ 文件存在: {topic_graph_path}")
    
    file_size = topic_graph_path.stat().st_size / (1024 * 1024)  # MB
    print(f"   文件大小: {file_size:.2f} MB")

    # 2. 加载文件
    try:
        with open(topic_graph_path, 'rb') as f:
            topic_graph = pickle.load(f)
        print(f"✅ 文件加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

    # 3. 检查数据类型
    if not isinstance(topic_graph, dict):
        print(f"❌ 数据类型错误: 期望 dict, 实际 {type(topic_graph)}")
        return False
    print(f"✅ 数据类型正确: {type(topic_graph)}")

    # 4. 检查数据量
    num_entities = len(topic_graph)
    print(f"✅ 实体数量: {num_entities:,}")

    # 5. 检查样本数据
    sample_entities = list(topic_graph.keys())[:3]
    print(f"\n样本实体 (前3个):")
    for entity in sample_entities:
        triple_ids = topic_graph[entity]
        entity_display = entity[:60] + "..." if len(entity) > 60 else entity
        print(f"  - 实体: {entity_display}")
        print(f"    关联三元组ID数量: {len(triple_ids):,}")
        if len(triple_ids) > 0:
            print(f"    前3个ID: {triple_ids[:3]}")

    # 6. 验证ID是否在triple2id范围内
    if triple2id_path.exists():
        print(f"\n验证ID范围...")
        with open(triple2id_path, 'rb') as f:
            triple2id = pickle.load(f)
        max_id = max(triple2id.values()) if triple2id else -1
        print(f"✅ triple2id 最大ID: {max_id:,}")
        
        # 检查是否有超出范围的ID
        all_ids = []
        for ids in topic_graph.values():
            all_ids.extend(ids)
        if all_ids:
            max_used_id = max(all_ids)
            min_used_id = min(all_ids)
            print(f"✅ topic_graph 使用的ID范围: {min_used_id:,} ~ {max_used_id:,}")
            if max_used_id > max_id:
                print(f"⚠️  警告: 存在超出范围的ID ({max_used_id} > {max_id})")
                return False
            else:
                print(f"✅ 所有ID都在有效范围内")
    else:
        print(f"⚠️  未找到 triple2id 文件，跳过ID范围验证")

    # 7. 统计信息
    total_triple_ids = sum(len(ids) for ids in topic_graph.values())
    avg_triples_per_entity = total_triple_ids / num_entities if num_entities > 0 else 0
    print(f"\n统计信息:")
    print(f"  - 总三元组ID引用数: {total_triple_ids:,}")
    print(f"  - 平均每个实体关联的三元组数: {avg_triples_per_entity:.2f}")
    
    # 8. 检查是否有空实体
    empty_entities = [k for k, v in topic_graph.items() if len(v) == 0]
    if empty_entities:
        print(f"⚠️  警告: 有 {len(empty_entities)} 个实体没有关联三元组")
    else:
        print(f"✅ 所有实体都有关联的三元组")

    print("\n" + "=" * 60)
    print("✅ 验证完成！文件格式正确。")
    print("=" * 60)
    return True

if __name__ == '__main__':
    verify_topic_graph()

