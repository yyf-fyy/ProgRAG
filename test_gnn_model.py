#!/usr/bin/env python3
"""
测试 GNN 模型的独立脚本
用于验证 GNN 模型是否能正常工作
"""
import torch
from pathlib import Path
import pickle
import json
from GNN.gnn_test import test
from GNN.nbfmodels import GNNRetriever, QueryNBFNet
from GNN.gnn_utils import make_gnn_first_input, make_gnn_second_input
from GNN.gnn_text_encoder import GTELargeEN

def test_gnn_model(dataset='cwq', device='cuda:0'):
    """
    测试 GNN 模型的基本功能
    
    Args:
        dataset: 数据集名称 ('cwq' 或 'webqsp')
        device: 使用的设备
    """
    print("=" * 60)
    print(f"测试 {dataset.upper()} 数据集的 GNN 模型")
    print("=" * 60)
    
    # 1. 加载 GNN 模型
    print("\n1. 加载 GNN 模型...")
    gnn_model_path = Path(f'ckpt/GNN/{dataset}/GNN.pth')
    if not gnn_model_path.exists():
        print(f"❌ 模型文件不存在: {gnn_model_path}")
        return False
    
    if dataset == 'webqsp':
        gnn_model = GNNRetriever(
            entity_model=QueryNBFNet(input_dim=512, hidden_dims=[512, 512, 512]), 
            rel_emb_dim=1024
        )
    else:  # cwq
        gnn_model = GNNRetriever(
            entity_model=QueryNBFNet(input_dim=512, hidden_dims=[512, 512, 512, 512, 512, 512]), 
            rel_emb_dim=1024
        )
    
    state = torch.load(gnn_model_path, map_location='cpu')
    if 'model' in state:
        gnn_model.load_state_dict(state['model'])
    else:
        gnn_model.load_state_dict(state)
    gnn_model.to(device)
    gnn_model.eval()
    print(f"✅ 模型加载成功，已移动到 {device}")
    
    # 2. 加载测试数据
    print("\n2. 准备测试数据...")
    
    # 加载 topic_graph 和 triple2id
    topic_graph_path = Path(f'data/graphs/{dataset}_topic_graph.pickle')
    triple2id_path = Path(f'data/graphs/{dataset}_triple2id.pickle')
    
    if not topic_graph_path.exists():
        print(f"❌ topic_graph 文件不存在: {topic_graph_path}")
        return False
    
    if not triple2id_path.exists():
        print(f"❌ triple2id 文件不存在: {triple2id_path}")
        return False
    
    with open(topic_graph_path, 'rb') as f:
        topic_graphs = pickle.load(f)
    
    with open(triple2id_path, 'rb') as f:
        triple2id = pickle.load(f)
    
    id2triple = {v: k for k, v in triple2id.items()}
    print(f"✅ 加载了 {len(topic_graphs)} 个主题实体的子图")
    
    # 3. 选择一个测试实体
    print("\n3. 选择测试实体...")
    test_entities = [k for k in topic_graphs.keys() if not k.startswith('g.') and not k.startswith('m.')][:5]
    if not test_entities:
        print("❌ 没有找到合适的测试实体")
        return False
    
    test_entity = test_entities[0]
    print(f"✅ 选择测试实体: {test_entity}")
    
    # 4. 构建知识图谱数据
    print("\n4. 构建知识图谱数据...")
    entity_subgraph = topic_graphs[test_entity]
    kg_data = make_gnn_first_input(entity_subgraph, id2triple)
    kg_data = kg_data.to(device)
    
    # 加载关系嵌入
    from GNN.gnn_utils import get_emb
    kg_data['rel_emb'] = get_emb(kg_data, dataset).to(device)
    print(f"✅ 知识图谱数据构建完成")
    print(f"   节点数: {kg_data.num_nodes}")
    print(f"   边数: {kg_data.edge_index.shape[1]}")
    print(f"   关系数: {kg_data.num_relations}")
    
    # 5. 构建问题数据
    print("\n5. 构建测试问题...")
    text_encoder = GTELargeEN(device)
    test_question = f"What is related to {test_entity}?"
    target_entities = list(topic_graphs.keys())[:10]  # 选择一些目标实体
    
    question_dataset = make_gnn_second_input(
        test_question, 
        test_entity, 
        [test_entity], 
        target_entities, 
        text_encoder, 
        kg_data, 
        encode_path=False
    )
    print(f"✅ 测试问题构建完成: {test_question}")
    
    # 6. 运行测试
    print("\n6. 运行 GNN 推理...")
    try:
        with torch.no_grad():
            selected_target, entity2prob = test(gnn_model, kg_data, question_dataset, device=device)
        
        print(f"✅ 推理成功！")
        print(f"\n预测结果 (Top 10):")
        for i, (entity, prob) in enumerate(list(entity2prob.items())[:10]):
            print(f"  {i+1}. {entity[:50]:50s} (概率: {prob:.4f})")
        
        print("\n" + "=" * 60)
        print("✅ GNN 模型测试通过！模型工作正常。")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'cwq'
    device = sys.argv[2] if len(sys.argv) > 2 else 'cuda:0'
    test_gnn_model(dataset, device)

