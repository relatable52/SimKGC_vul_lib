import json
import os
import pickle
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer

class DataEntry:
    def __init__(
        self, *,
        head_id: str, head: str,
        tail_id: str, tail: str,
        relation: str
    ):
        self.head = head
        self.head_id = head_id
        self.tail = tail
        self.tail_id = tail_id
        self.relation = relation

class KGDataset(Dataset):
    def __init__(
        self, *,
        data_path: str,
        entities_path: str,
        relations_path: str,
        neighbor_path: str,
        cache_dir: str,
        name: str = '',
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512
    ):
        assert data_path.endswith('.json'), f'Unsupported format: {data_path}'
        self.data_path = data_path
        self.entities_path = entities_path
        self.relations_path = relations_path
        self.neighbor_path = neighbor_path
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.name = os.path.basename(self.data_path) #File name for logging
        file_name = os.path.splitext(os.path.basename(self.data_path))[0]
        self.cache_path = os.path.join(self.cache_dir, name + '_' + file_name + '.pkl')

        self._build_entities_dict()
        self._build_neighbor_dict()

        self.data = {
            'triplet': [],
            'hr_token_ids': [],
            'hr_token_type_ids': [],
            'hr_mask': [],
            'tail_token_ids': [],
            'tail_token_type_ids': [],
            'tail_mask': [],
            'head_token_ids': [],
            'head_token_type_ids': [],
            'head_mask': [],
        }
        self.keys = self.data.keys()

        if not os.path.exists(self.cache_path):
            self._preprocess()
            self._save_cache()
        else:
            self._load()

    def __len__(self):
        return len(self.data['hr_token_ids'])

    def __getitem__(self, index):
        item = {key: self.data[key][index] for key in self.keys}
        return item
    
    def _preprocess(self):
        json_data = self._load_json_data()

        for entry in (loop := tqdm(json_data)):
            loop.set_description('Processing dataset: ' + self.name)
            tokenized_inputs = self._tokenize(entry=entry)
            for key in self.keys:
                self.data[key].append(tokenized_inputs[key])
        
    def _save_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.data, f)
    
    def _load(self):
        with open(self.cache_path, 'rb') as f:
            self.data = pickle.load(f)

    def _load_json_data(self) -> list[DataEntry]:
        with open(self.data_path, 'r', encoding='utf8') as f:
            json_data = json.load(f)
        data = [DataEntry(**entry) for entry in json_data]
        return data
    
    def _build_entities_dict(self):
        with open(self.entities_path, 'r', encoding='utf8') as f:
            entities = json.load(f)
        self.entity2id = {}
        for index, entity in enumerate(entities):
            self.entity2id[entity['entity_id']] = index

        with open(self.relations_path, 'r', encoding='utf8') as f:
            relations = json.load(f)
        for index, relation in enumerate(list(relations.values())):
            self.entity2id[relation] = index
            print(relation, index)

    def _build_neighbor_dict(self):
        with open(self.neighbor_path, 'r', encoding='utf8') as f:
            relations = json.load(f)
        self.neighbor_dict = defaultdict(set)
        for entry in relations:
            hr = (self.entity2id[entry['head_id']], self.entity2id[entry['relation']])
            tail = self.entity2id[entry['tail_id']]
            self.neighbor_dict[hr].add(tail)

    def _tokenize(self, entry: DataEntry):
        tokenize_setting = {
            'add_special_tokens': True,
            'max_length': self.max_length,
            'return_token_type_ids': True,
            'truncation': True,
            'padding': 'max_length',
            'return_tensors': 'pt'
        }

        head = entry.head
        tail = entry.tail
        relation = entry.relation
        triplet = [
            self.entity2id[entry.head_id],
            self.entity2id[entry.relation],
            self.entity2id[entry.tail_id]
        ]

        hr_inputs = self.tokenizer(text=head, text_pair=relation, **tokenize_setting)
        tail_inputs = self.tokenizer(text=tail, **tokenize_setting)
        head_inputs = self.tokenizer(text=head, **tokenize_setting)

        return {
            'triplet': triplet,
            'hr_token_ids': hr_inputs['input_ids'],
            'hr_token_type_ids': hr_inputs['token_type_ids'],
            'hr_mask': hr_inputs['attention_mask'],
            'tail_token_ids': tail_inputs['input_ids'],
            'tail_token_type_ids': tail_inputs['token_type_ids'],
            'tail_mask': tail_inputs['attention_mask'],
            'head_token_ids': head_inputs['input_ids'],
            'head_token_type_ids': head_inputs['token_type_ids'],
            'head_mask': head_inputs['attention_mask'],
        }
    
    def collate(self, batch_data: list[dict]) -> dict:
        batch_dict = {
            'hr_token_ids': None,
            'hr_mask': None,
            'hr_token_type_ids': None,
            'tail_token_ids': None,
            'tail_mask': None,
            'tail_token_type_ids': None,
            'head_token_ids': None,
            'head_mask': None,
            'head_token_type_ids': None
        }

        triplet = [entry['triplet'] for entry in batch_data]
        inbatch_mask = self._inbatch_negatives_mask(triplet)
        self_mask = self._self_negatives_mask(triplet)

        for key in batch_dict.keys():
            batch_dict[key] = torch.stack([entry[key] for entry in batch_data])

        batch_dict['triplet_mask'] = inbatch_mask
        batch_dict['self_negative_mask'] = self_mask
        
        return batch_dict
    
    def _inbatch_negatives_mask(self, triplet: torch.Tensor) -> torch.Tensor:
        mask = [
            [
                ((i[2] != j[2]) and (j[2] not in self.neighbor_dict[(i[0],i[1])])) for j in triplet
            ]
            for i in triplet
        ]
        mask = torch.tensor(mask)
        mask.fill_diagonal_(True)
        return mask
    
    def _self_negatives_mask(self, triplet: torch.Tensor) -> torch.Tensor:
        mask = [
            (i[0] in self.neighbor_dict[(i[0], i[1])]) for i in triplet
        ]
        mask = torch.tensor(mask)
        return mask
    
if __name__ == '__main__':
    train = KGDataset(
        data_path='data/test.txt.json',
        entities_path='data/entities.json',
        relations_path='data/relations.json',
        neighbor_path='data/neighbor.json',
        cache_dir='data/cache',
        name = 'vul_lib',
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased'),
        max_length=5
    )
    train_loader = DataLoader(
        train, 
        batch_size=3,
        collate_fn=train.collate
    )
    batch = next(iter(train_loader))
    print(batch)