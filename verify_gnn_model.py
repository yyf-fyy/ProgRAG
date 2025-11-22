#!/usr/bin/env python3
"""
验证 GNN 模型和相关文件是否准备就绪
"""
import torch
from pathlib import Path
import pickle

def verify_gnn_model(dataset='cwq'):
    print("=" * 60)
    print(f"验证 {dataset.upper()} 数据集的 GNN 模型和相关文件")
    print("=" * 60)
    
    all_ok = True
    
    # 1. 检查 GNN 模型文件
    gnn_model_path = Path(f'ckpt/GNN/{dataset}/GNN.pth')
    print(f"\n1. 检查 GNN 模型文件...")
    if gnn_model_path.exists():
        try:
            state = torch.load(gnn_model_path, map_location='cpu')
            print(f"   ✅ 文件存在: {gnn_model_path}")
            print(f"   文件大小: {gnn_model_path.stat().st_size / (1024**2):.2f} MB")
            
            # 检查状态字典结构
            if 'model' in state:
                model_state = state['model']
                print(f"   ✅ 包含模型权重，参数数量: {len(model_state)}")
                # 显示一些参数形状
                sample_keys = list(model_state.keys())[:3]
                for key in sample_keys:
                    if hasattr(model_state[key], 'shape'):
                        print(f"      - {key}: {model_state[key].shape}")
            else:
                print(f"   ⚠️  状态字典中没有 'model' 键，直接包含参数")
                print(f"   参数数量: {len(state)}")
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            all_ok = False
    else:
        print(f"   ❌ 文件不存在: {gnn_model_path}")
        all_ok = False
    
    # 2. 检查关系嵌入文件
    rel_emb_path = Path(f'data/{dataset}/emb/relation.pth')
    print(f"\n2. 检查关系嵌入文件...")
    if rel_emb_path.exists():
        try:
            rel_emb = torch.load(rel_emb_path, map_location='cpu')
            if isinstance(rel_emb, dict):
                print(f"   ✅ 文件存在: {rel_emb_path}")
                print(f"   文件大小: {rel_emb_path.stat().st_size / (1024**2):.2f} MB")
                print(f"   关系数量: {len(rel_emb)}")
                # 检查嵌入维度
                sample_rel = list(rel_emb.keys())[0]
                emb_dim = rel_emb[sample_rel].shape[0] if hasattr(rel_emb[sample_rel], 'shape') else len(rel_emb[sample_rel])
                print(f"   嵌入维度: {emb_dim}")
                print(f"   样本关系: {sample_rel[:50]}...")
            else:
                print(f"   ⚠️  文件格式不是字典: {type(rel_emb)}")
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            all_ok = False
    else:
        print(f"   ❌ 文件不存在: {rel_emb_path}")
        print(f"   💡 提示: 需要运行 'python GNN/get_emb.py -d {dataset} --graph_file data/graphs/total_graph_{dataset}.jsonl --device cuda:0'")
        all_ok = False
    
    # 3. 检查映射文件
    entity2id_path = Path(f'data/{dataset}/emb/entity2id.pkl')
    rel2id_path = Path(f'data/{dataset}/emb/rel2id.pkl')
    
    print(f"\n3. 检查映射文件...")
    if entity2id_path.exists():
        try:
            with open(entity2id_path, 'rb') as f:
                entity2id = pickle.load(f)
            print(f"   ✅ entity2id.pkl 存在，实体数量: {len(entity2id):,}")
        except Exception as e:
            print(f"   ❌ entity2id.pkl 加载失败: {e}")
            all_ok = False
    else:
        print(f"   ⚠️  entity2id.pkl 不存在（可能由 get_emb.py 自动生成）")
    
    if rel2id_path.exists():
        try:
            with open(rel2id_path, 'rb') as f:
                rel2id = pickle.load(f)
            print(f"   ✅ rel2id.pkl 存在，关系数量: {len(rel2id):,}")
        except Exception as e:
            print(f"   ❌ rel2id.pkl 加载失败: {e}")
            all_ok = False
    else:
        print(f"   ⚠️  rel2id.pkl 不存在（可能由 get_emb.py 自动生成）")
    
    # 4. 尝试加载模型（如果所有文件都存在）
    if all_ok:
        print(f"\n4. 尝试加载 GNN 模型...")
        try:
            from GNN.nbfmodels import GNNRetriever, QueryNBFNet
            
            # 根据数据集选择模型结构
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
            
            print(f"   ✅ 模型加载成功！")
            print(f"   模型参数总数: {sum(p.numel() for p in gnn_model.parameters()):,}")
            
        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ 所有检查通过！GNN 模型已准备就绪。")
        print("\n下一步可以：")
        print("  1. 运行推理: python main.py --dataset cwq")
        print("  2. 或测试模型: python GNN/gnn_test.py")
    else:
        print("⚠️  部分文件缺失或有问题，请检查上述提示。")
    print("=" * 60)
    
    return all_ok

if __name__ == '__main__':
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'cwq'
    verify_gnn_model(dataset)

