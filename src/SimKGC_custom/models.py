from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn

POOLING_TYPE = ['cls', 'max', 'mean']

class CustomEncoder(nn.Module):
    def __init__(
        self, *,
        pretrained_model: str,
        pooling: str = 'cls',
        **kwargs
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        self.hr_encoder = EntitiesEncoder(pretrained_model=pretrained_model, pooling=pooling)
        self.tail_encoder = EntitiesEncoder(pretrained_model=pretrained_model, pooling=pooling)

    def forward(
        self,
        hr_token_ids: torch.Tensor, hr_mask: torch.Tensor, hr_token_type_ids: torch.Tensor,
        tail_token_ids: torch.Tensor, tail_mask: torch.Tensor, tail_token_type_ids: torch.tensor,
        head_token_ids: torch.Tensor, head_mask: torch.Tensor, head_token_type_ids: torch.Tensor,
        **kwargs
    ):
        
        hr_vector = self.hr_encoder(
            input_ids = hr_token_ids,
            attention_mask = hr_mask,
            token_type_ids = hr_token_type_ids
        )

        tail_vector = self.tail_encoder(
            input_ids = tail_token_ids,
            attention_mask = tail_mask,
            token_type_ids = tail_token_type_ids
        )

        head_vector = self.tail_encoder(
            input_ids = head_token_ids,
            attention_mask = head_mask,
            token_type_ids = head_token_type_ids
        )

        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}
    
    def predict_entity(
        self, *,
        input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.tensor,
        **kwargs
    ):
        outputs = self.tail_encoder(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        return outputs

class EntitiesEncoder(nn.Module):
    def __init__(
        self, *,
        pretrained_model: str,
        pooling: str
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
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
        self, *,
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
        cls_output = self._pool_output(
            pooling=self.pooling, 
            cls_output=cls_output,
            mask=attention_mask,
            last_hidden_state=last_hidden_state
        )
        return cls_output
        