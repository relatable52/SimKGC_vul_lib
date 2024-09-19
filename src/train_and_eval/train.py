import json
from tqdm import tqdm

from src.SimKGC_custom.dataset import KGDataset
from src.SimKGC_custom.models import CustomEncoder
from src.SimKGC_custom.trainer import KGCTrainer, LoaderSetting
from src.train_and_eval.logger import logger

from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    max_length = 256
    train = KGDataset(
        data_path='data/train.txt.json',
        entities_path='data/entities.json',
        relations_path='data/relations.json',
        neighbor_path='data/neighbor.json',
        cache_dir='data/cache',
        name = 'vul_lib',
        tokenizer = tokenizer,
        max_length=max_length
    )
    valid = KGDataset(
        data_path='data/valid.txt.json',
        entities_path='data/entities.json',
        relations_path='data/relations.json',
        neighbor_path='data/neighbor.json',
        cache_dir='data/cache',
        name = 'vul_lib',
        tokenizer = tokenizer,
        max_length=max_length
    )

    model = CustomEncoder(pretrained_model='bert-base-cased', pooling='max')
    setting = LoaderSetting(
        batch_size = 256,
        num_workers = 2
    )

    trainer = KGCTrainer(
        model = model, 
        train = train,
        save_dir = 'checkpoints',
        logger = logger,
        test = valid, 
        train_setting = setting
    )
    trainer.train(epochs = 15)
        
if __name__ == "__main__":
    main()
