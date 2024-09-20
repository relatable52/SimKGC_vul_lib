import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomInfoNCELoss(nn.Module):
    def __init__(self, *, temp: float, margin: float, use_self_negative: bool=False):
        super().__init__()
        self.temp = temp
        self.margin = margin
        self.use_self_negative = use_self_negative

    def forward(
        self,
        hr_vector: torch.Tensor,
        tail_vector: torch.Tensor,
        head_vector: torch.Tensor,
        triplet_mask: torch.Tensor,
        self_negative_mask: torch.Tensor=None
    ):
        return infonce(
            hr_vector=hr_vector,
            tail_vector=tail_vector,
            head_vector=head_vector,
            triplet_mask=triplet_mask,
            self_negative_mask=self_negative_mask,
            temp=self.temp,
            margin=self.margin,
            use_self_negative=self.use_self_negative
        )

def infonce(
    hr_vector: torch.Tensor,
    tail_vector: torch.Tensor,
    head_vector: torch.Tensor,
    triplet_mask: torch.Tensor,
    self_negative_mask: torch.Tensor,
    use_self_negative: bool,
    temp: float,
    margin: float
):
    norm_hr_vector = F.normalize(hr_vector, p=2, dim=1)
    norm_tail_vector = F.normalize(tail_vector, p=2, dim=1)
    norm_head_vector = F.normalize(head_vector, p=2, dim=1)

    logits = F.cosine_similarity(norm_hr_vector, norm_tail_vector, dim=1)
    batch_size = hr_vector.size(0)
    labels = torch.arange(batch_size).to(hr_vector.device)
    logits -= torch.zeros(logits.size()).fill_diagonal_(margin).to(logits.device)
    logits /= temp

    logits.masked_fill_(~triplet_mask, -1e6)

    if use_self_negative:
        self_neg_logits = torch.sum(norm_hr_vector*norm_head_vector, dim=1)/temp
        self_neg_logits.masked_fill_(~self_negative_mask, -1e6)
        logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

    loss = F.cross_entropy(logits, labels)
    return loss