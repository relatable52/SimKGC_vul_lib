import json
import os
from dataclasses import dataclass
from logging import Logger, getLogger
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, LambdaLR

from .models import CustomEncoder
from .dataset import KGDataset
from .metric import accuracy_at_k
from .infonce import CustomInfoNCELoss

def move_to_cuda(sample):
    """A helper function to move batches to gpu"""
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

@dataclass
class LoaderSetting:
    batch_size: int = 256
    num_workers: int = 2

class KGCTrainer:
    def __init__(
        self, *,
        model: CustomEncoder,
        train: KGDataset,
        save_dir: str,
        train_setting: LoaderSetting = LoaderSetting(),
        logger: Logger = None,
        test: KGDataset = None,
        test_setting: LoaderSetting = None,
        use_amp = False
    ):
        general_loader_setting = {
            'shuffle': True,
            'pin_memory': True,
            'drop_last': True
        }

        self.model = model

        self.train_loader = DataLoader(
            train, 
            collate_fn=train.collate,
            batch_size=train_setting.batch_size,
            num_workers=train_setting.num_workers,
            **general_loader_setting
        )

        if test:
            setting = test_setting if test_setting else train_setting
            self.test_loader = DataLoader(
                test,
                collate_fn=test.collate,
                batch_size=setting.batch_size,
                num_workers=setting.num_workers,
                **general_loader_setting
            )

        self.logger = logger if logger else getLogger()
        self.use_amp = use_amp
        self.save_dir = save_dir

    def train(
        self, 
        epochs: int, 
        use_self_negative = True,
        temp: float = 0.05, 
        margin: float = 0.05,
        learning_rate: float = 5e-5,
        weight_decay: float = 1e-4,
        print_freq: int = 10
    ):
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr = learning_rate,
            weight_decay = weight_decay
        )

        self.criterion = CustomInfoNCELoss(
            temp=temp,
            margin=margin,
            use_self_negative=use_self_negative
        )

        if self.use_amp:
            self.scaler = torch.amp.GradScaler()

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            self.logger.info('No gpu will be used')

        accuracy = 0
        for epoch in range(epochs):
            self.train_epoch(epoch=epoch, print_freq=print_freq)
            temp_accuracy = self._eval(epoch=epoch)
            if temp_accuracy > accuracy: 
                accuracy = temp_accuracy
                self.save_model('best.pth')
        self.save_model('last.pth')

    def save_model(self, filename):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, filename))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, filename))

    def train_epoch(self, epoch: int, print_freq: int):
        losses = AverageMeter('Loss', ':.4')
        batches = len(self.train_loader)
        for i, batch_dict in (loop := tqdm(enumerate(self.train_loader))):
            self.model.train()
        
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            
            batch_size = batch_dict['hr_token_ids'].size(0)

            if self.use_amp:
                with torch.cuda.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)

            loss = self.criterion(
                hr_vector = outputs['hr_vector'],
                tail_vector = outputs['tail_vector'],
                head_vector = outputs['head_vector'],
                triplet_mask = batch_dict['triplet_mask'],
                self_negative_mask = batch_dict['self_negative_mask']
            )

            losses.update(loss.item(), batch_size)

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if i % print_freq == 0:
                loop.set_description(f'Epoch{epoch} - {i}/{batches}')
                loop.set_postfix({'loss':round(losses.val, 3)})
        self.logger.info(f'Epoch {epoch}: Loss={round(losses.avg, 3)}')

    @torch.no_grad()
    def _eval(self, epoch):
        if not self.test_loader:
            return 0
        
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        for i, batch_dict in tqdm(enumerate(self.test_loader)):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            
            batch_size = batch_dict['hr_token_ids'].size(0)

            outputs = self.model(**batch_dict)

            loss = self.criterion(
                hr_vector = outputs['hr_vector'],
                tail_vector = outputs['tail_vector'],
                head_vector = outputs['head_vector'],
                triplet_mask = batch_dict['triplet_mask'],
                self_negative_mask = batch_dict['self_negative_mask']
            )

            logits = outputs['hr_vector'].mm(outputs['tail_vector'].t())
            labels = torch.arange(batch_size).to(outputs['hr_vector'].device)

            acc1 = accuracy_at_k(logits=logits, labels=labels, k=1)
            acc5 = accuracy_at_k(logits=logits, labels=labels, k=5)

            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)
        
        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top5.avg, 3),
                       'loss': round(losses.avg, 3)}
        self.logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        return top1.avg


        
