import torch
import json
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser

from src.SimKGC.trainer import Trainer
from src.train_and_eval.logger import logger
from src.train_and_eval.utils import read_config

def get_args(config_path):
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src\train_and_eval\config\config.json")
    return parser.parse_args()

def main():
    args = get_args()
    config = read_config(args.config_path)

    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    logger.info("Use {} gpus for training".format(ngpus_per_node))

    trainer = Trainer(config, ngpus_per_node=ngpus_per_node)
    logger.info('Args={}'.format(json.dumps(config.__dict__, ensure_ascii=False, indent=4)))
    trainer.train_loop()

if __name__ == '__main__':
    main()
