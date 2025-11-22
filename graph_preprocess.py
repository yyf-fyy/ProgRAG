import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / 'data'
GRAPHS_DIR = DATA_ROOT / 'graphs'
DATA_ROOT.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

def make_total_graph(dataset):
    # dataset = 'cwq'

    train_dataset = load_dataset(f"rmanluo/RoG-{dataset}", split='train')
    valid_dataset = load_dataset(f"rmanluo/RoG-{dataset}", split='validation')
    test_dataset = load_dataset(f"rmanluo/RoG-{dataset}", split='test')

    temp_set = set()
    for item in train_dataset:
        for t in item['graph']:
            temp_set.add(tuple(t))

    for item in valid_dataset:
        for t in item['graph']:
            temp_set.add(tuple(t))

    for item in test_dataset:
        for t in item['graph']:
            temp_set.add(tuple(t))

    temp_list = list(temp_set)
    final_list = []
    for t in temp_list:
        final_list.append(list(t))

    with open(GRAPHS_DIR / f'total_graph_{dataset}.jsonl', 'w', encoding='utf-8') as file:
        json.dump(final_list, file)



def make_topic2graph(dataset):
    train_dataset = load_dataset(f"rmanluo/RoG-{dataset}", split='train')
    valid_dataset = load_dataset(f"rmanluo/RoG-{dataset}", split='validation')
    test_dataset = load_dataset(f"rmanluo/RoG-{dataset}", split='test')

    topic2graph = dict()

    split_datasets = [
        ('train', train_dataset),
        ('validation', valid_dataset),
        ('test', test_dataset),
    ]

    for split_name, split_dataset in split_datasets:
        for item in tqdm(split_dataset, desc=f'构建topic图 ({split_name})'):
            q_list = item['q_entity']
            subgraph = item['graph']
            temp_graph = set()
            for triple in subgraph:
                temp_graph.add(tuple(triple))

            for entity in q_list:
                if entity not in topic2graph:
                    topic2graph[entity] = set()
                topic2graph[entity] |= temp_graph

    topic2graph = {k : list(v) for k, v in topic2graph.items()}

    triple2id_candidates = [
        GRAPHS_DIR / f'{dataset}_triple2id.pkl',
        GRAPHS_DIR / f'{dataset}_triple2id.pickle',
        DATA_ROOT / dataset / f'{dataset}_triple2id.json',
    ]
    triple2id = None
    for path in triple2id_candidates:
        if path.exists():
            if path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    triple2id = json.load(f)
            else:
                with open(path, 'rb') as f:
                    triple2id = pickle.load(f)
            break
    if triple2id is None:
        raise FileNotFoundError(
            f"Could not find triple2id file for dataset '{dataset}'. "
            f"Expected one of: {[str(p) for p in triple2id_candidates]}"
        )

    cwq_graph = dict()
    for k, v in topic2graph.items():
        if k not in cwq_graph:
            cwq_graph[k] = list()
        cwq_graph[k] += [triple2id[triplet] for triplet in v]
        
    with open(GRAPHS_DIR / f'{dataset}_topic_graph.pickle', mode='wb') as f:
        pickle.dump(cwq_graph, f)

dataset = 'cwq'
make_total_graph(dataset)
make_topic2graph(dataset)
