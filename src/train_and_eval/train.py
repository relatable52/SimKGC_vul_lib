import torch
import json
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser

from src.SimKGC.trainer import Trainer
from .logger import logger
from src.SimKGC.config import args

def main():
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    logger.info("Use {} gpus for training".format(ngpus_per_node))

    trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.train_loop()

if __name__ == '__main__':
    main()
