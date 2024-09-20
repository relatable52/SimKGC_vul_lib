import networkx as nx
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from node2vec import Node2Vec

import torch

from transformers import PreTrainedTokenizer
from src.SimKGC_custom.models import CustomEncoder
from src.SimKGC_custom.trainer import move_to_cuda

def add_desc_embeddings(
    graph: nx.DiGraph,
    model: CustomEncoder,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512 
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for node in (loop:=tqdm(graph.nodes(data=True))):
        loop.set_description('Adding embeddings')
        node_id, node_data = node
        description = node_data['description']

        inputs = tokenizer(
            description, 
            return_tensors='pt', 
            add_special_tokens=True, return_token_type_ids=True, truncation=True, padding=True, 
            max_length=max_length
        )

        inputs = move_to_cuda(inputs)

        with torch.no_grad():
            outputs = model.predict_entity(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

        graph.nodes[node_id]['embedding'] = embedding
    
    return graph

def add_node2vec_embeddings(graph, dimensions=64, walk_length=30, num_walks=200, p=1, q=1, workers=4):
    # Step 1: Create Node2Vec model
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, 
                        num_walks=num_walks, p=p, q=q, workers=workers)
    
    # Step 2: Fit the model to learn embeddings
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # Step 3: Add Node2Vec embeddings as node attributes
    for node in (loop := tqdm(graph.nodes())):
        loop.set_description('Adding node2vec')
        # Get the embedding for the node (model expects node ids as strings)
        embedding = model.wv[str(node)]  # Convert node ID to string if necessary
        
        # Add embedding as a node attribute
        graph.nodes[node]['node2vec_embedding'] = np.array(embedding)
    
    return graph

def build_graph(
    entities_path: str,
    neighbor_path: str
):
    graph = nx.DiGraph()

    with open(entities_path, 'r', encoding='utf8') as f:
        entities = pd.DataFrame(json.load(f))
    
    with open(neighbor_path, 'r', encoding='utf8') as f:
        neighbors = pd.DataFrame(json.load(f))

    for _, row in entities.iterrows():
        entity_id = row['entity_id']
        description = row['entity_id']+row['entity']
        graph.add_node(entity_id, description=description)

    for _, row in neighbors.iterrows():
        head_id = row['head_id']
        tail_id = row['tail_id']
        relation = row['relation']
        graph.add_edge(head_id, tail_id, relation_type=relation)

    return graph

if __name__ == "__main__":
    graph = build_graph('data/entities.json', 'data/neighbor.json')
    

    
