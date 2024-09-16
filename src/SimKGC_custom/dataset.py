import json
import os
import pickle
from tqdm import tqdm
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
        cache_dir: str,
        name: str = '',
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512
    ):
        assert data_path.endswith('.json'), f'Unsupported format: {data_path}'
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.name = os.path.basename(self.data_path) #File name for logging
        file_name = os.path.splitext(os.path.basename(self.data_path))[0]
        self.cache_path = os.path.join(self.cache_dir, name + file_name + '.pkl')

        self.data = {
            'hr_token_ids': [],
            'hr_token_type_ids': [],
            'tail_token_ids': [],
            'tail_token_type_ids': [],
            'head_token_ids': [],
            'head_token_type_ids': [],
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
    
    def _tokenize(self, entry: DataEntry):
        tokenize_setting = {
            'add_special_tokens': True,
            'max_length': self.max_length,
            'return_token_type_ids': True,
            'truncation': True,
            'padding': 'max_length'
        }

        head = entry.head
        tail = entry.tail
        relation = entry.relation

        hr_inputs = self.tokenizer(text=head, text_pair=relation, **tokenize_setting)
        tail_inputs = self.tokenizer(text=tail, **tokenize_setting)
        head_inputs = self.tokenizer(text=head, **tokenize_setting)

        return {
            'hr_token_ids': hr_inputs['input_ids'],
            'hr_token_type_ids': hr_inputs['token_type_ids'],
            'tail_token_ids': tail_inputs['input_ids'],
            'tail_token_type_ids': tail_inputs['token_type_ids'],
            'head_token_ids': head_inputs['input_ids'],
            'head_token_type_ids': head_inputs['token_type_ids'],
        }
    
if __name__ == '__main__':
    train = KGDataset(
        data_path='data/test.txt.json',
        cache_dir='data/cache',
        name = 'vul_lib',
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased'),
        max_length=10
    )
    train_loader = DataLoader(
        train, 
        batch_size=1
    )
    batch = next(iter(train_loader))
    print(batch)