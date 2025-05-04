import gymnasium as gym
import math
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from alphagen.data.expression import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        "x: ([batch_size, ]seq_len, embedding_dim)"
        seq_len = x.size(0) if x.dim() == 2 else x.size(1)
        return x + self._pe[:seq_len]  # type: ignore


class TransformerSharedNet(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_encoder_layers: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        device: torch.device
    ):
        super().__init__(observation_space, d_model)

        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0] + 1                   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        self._transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head,
                dim_feedforward=d_ffn, dropout=dropout,
                activation=lambda x: F.leaky_relu(x),               # type: ignore
                batch_first=True, device=device
            ),
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(d_model, eps=1e-5, device=device)
        )

    def forward(self, obs: Tensor) -> Tensor:
        bs, seqlen = obs.shape
        beg = torch.full((bs, 1), fill_value=self._n_actions, dtype=torch.long, device=obs.device)
        obs = torch.cat((beg, obs.long()), dim=1)
        pad_mask = obs == 0
        src = self._pos_enc(self._token_emb(obs))
        res = self._transformer(src, src_key_padding_mask=pad_mask)
        return res.mean(dim=1)


class LSTMSharedNet(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_layers: int,
        d_model: int,
        dropout: float,
        device: torch.device
    ):
        super().__init__(observation_space, d_model)

        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0] + 1                   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        self._lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, obs: Tensor) -> Tensor:
        bs, seqlen = obs.shape
        beg = torch.full((bs, 1), fill_value=self._n_actions, dtype=torch.long, device=obs.device)
        obs = torch.cat((beg, obs.long()), dim=1)
        real_len = (obs != 0).sum(1).max()
        src = self._pos_enc(self._token_emb(obs))
        res = self._lstm(src[:,:real_len])[0]
        return res.mean(dim=1)
import torch
from transformers import GPT2Model, GPT2Config

class GPT2SharedNet(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        d_model: int = 768,
        dropout: float = 0.1,
        device: torch.device = 'cuda',
        pretrained: bool = True,
        freeze_transformer: bool = True,
        train_layernorm: bool = True,
        train_positional: bool = True
    ):
        super().__init__(observation_space, d_model)
        
        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0] + 1
        
        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions
        
        if pretrained:
            self._gpt2 = GPT2Model.from_pretrained('gpt2').to(device)
            
            # 冻结Transformer块
            if freeze_transformer:
                for name, param in self._gpt2.named_parameters():
                    if 'wpe' in name and train_positional:  # 位置编码
                        param.requires_grad = True
                    elif 'ln' in name and train_layernorm:  # Layer Norm
                        param.requires_grad = True
                    else:  # 其他参数(attention和ffn)
                        param.requires_grad = False
            
            self._token_proj = nn.Linear(n_actions + 1, self._gpt2.config.vocab_size).to(device)
        else:
            config = GPT2Config(
                vocab_size=n_actions + 1,
                n_positions=512,
                n_embd=d_model,
                n_layer=6,
                n_head=12,
                dropout=dropout
            )
            self._gpt2 = GPT2Model(config).to(device)
        
    def forward(self, obs: Tensor) -> Tensor:
        bs, seqlen = obs.shape
        beg = torch.full((bs, 1), fill_value=self._n_actions, dtype=torch.long, device=obs.device)
        obs = torch.cat((beg, obs.long()), dim=1)
        
        if hasattr(self, '_token_proj'):
            obs_onehot = F.one_hot(obs, num_classes=int(self._n_actions) + 1).float()
            obs = self._token_proj(obs_onehot)
            obs = obs.argmax(dim=-1)
        
        attention_mask = (obs != 0).float()
        outputs = self._gpt2(
            input_ids=obs,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return outputs.last_hidden_state.mean(dim=1)

class Decoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_layers: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        device: torch.device
    ):
        super().__init__(observation_space, d_model)

        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0] + 1                   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        # Actually an encoder for now
        self._decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=d_ffn,
                dropout=dropout, batch_first=True, device=device
            ),
            n_layers,
            norm=nn.LayerNorm(d_model, device=device)
        )

    def forward(self, obs: Tensor) -> Tensor:
        batch_size = obs.size(0)
        begins = torch.full(size=(batch_size, 1), fill_value=self._n_actions,
                            dtype=torch.long, device=obs.device)
        obs = torch.cat((begins, obs.type(torch.long)), dim=1)      # (bs, len)
        pad_mask = obs == 0
        res = self._token_emb(obs)                                  # (bs, len, d_model)
        res = self._pos_enc(res)                                    # (bs, len, d_model)
        res = self._decoder(res, src_key_padding_mask=pad_mask)     # (bs, len, d_model)
        return res.mean(dim=1)                                      # (bs, d_model)
