from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn

POOLING_TYPE = ['cls', 'max', 'mean']

class CustomEncoder(nn.Module):
    def __init__(
        self, *,
        pretrained_model: str,
        batch_size: int,
        pre_batch: int,
        additive_margin: float,
        t: float,
        finetune_t: bool,
        pooling: str = 'cls',
        **kwargs
    ):
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0/t), requires_grad=finetune_t)

        self.add_margin = additive_margin
        self.batch_size = batch_size
        self.pre_batch = pre_batch

        config = AutoConfig.from_pretrained(pretrained_model)
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, config.hidden_size)
        self.register_buffer(
            "pre_batch_vectors",
            nn.functional.normalize(random_vector, dim=1),
            persistent=False
        )
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_encoder = EntitiesEncoder(pretrained_model=pretrained_model, pooling=pooling)
        self.tail_encoder = EntitiesEncoder(pretrained_model=pretrained_model, pooling=pooling)

    def forward(
        self,
        hr_token_ids: torch.tensor, hr_mask: torch.tensor, hr_token_type_ids: torch.tensor,
        tail_token_ids: torch.tensor, tail_mask: torch.tensor, tail_token_type_ids: torch.tensor,
        head_token_ids: torch.tensor, head_mask: torch.tensor, head_token_type_ids: torch.tensor,
        only_ent_embedding: bool = False,
        **kwargs
    ):
        if only_ent_embedding:
            return self.predict_ent_embedding(
                tail_token_ids=tail_token_ids,
                tail_mask = tail_mask,
                tail_token_type_ids=tail_token_type_ids
            )
        
        hr_vector = self.hr_encoder(
            input_ids = hr_token_ids,
            attetion_mask = hr_mask,
            token_type_ids = hr_token_type_ids
        )

        tail_vector = self.tail_encoder(
            input_ids = tail_token_ids,
            attetion_mask = tail_mask,
            token_type_ids = tail_token_type_ids
        )

        head_vector = self.tail_encoder(
            input_ids = head_token_ids,
            attetion_mask = head_mask,
            token_type_ids = head_token_type_ids
        )

        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}
    
    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}
    
    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self.tail_encoder(
            input_ids = tail_token_ids,
            attetion_mask = tail_mask,
            token_type_ids = tail_token_type_ids
        )
        return {'ent_vectors': ent_vectors.detach()}

class EntitiesEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model: str,
        pooling: str
    ):
        self.encoder = AutoModel.from_pretrained(pretrained_model),
        self.pooling = pooling
        assert pooling in POOLING_TYPE, f'Unknown pooling mode: {pooling}'
                 
    def _pool_output(
        self, 
        pooling: str,
        cls_output: torch.Tensor,
        mask: torch.Tensor,
        last_hidden_state: torch.Tensor
    ) -> torch.Tensor:
        if pooling == 'cls':
            output_vector = cls_output
        elif pooling == 'max':
            input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
            last_hidden_state[input_mask_expanded == 0] = -1e4
            output_vector = torch.max(last_hidden_state, 1)[0]
        elif pooling == 'mean':
            input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
            output_vector = sum_embeddings / sum_mask
        else:
            assert False, f'Unknown pooling mode: {pooling}'

        output_vector = nn.functional.normalize(output_vector, dim=1)
        return output_vector
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ):
        outputs = self.encoder(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            return_dict = True
        )

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = EntitiesEncoder._pool_output(
            pooling=self.pooling, 
            cls_output=cls_output,
            mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return cls_output