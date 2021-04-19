import torch
import torch.nn as nn
from modules.encoder import set_encoder
from modules.predictor import set_predictor
from modules.generator import set_decoder

class Wrapper(nn.Module):
    def __init__(self, encoder=None, predictor=None, generator=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.generator = generator

        # For gradient-based method:
        # find the last layer of encoder
        self.gradients = []
        for name, module in self.encoder.named_modules():
            pass 
        self.encoder_last_layer = name
        # hook
        self.encoder.register_backward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.gradients.append(output[0])

    def forward(self, batch):
        self.gradients = []
        batch = self.encoder(batch)

        # If Caption module exists: generate caption
        caption = self.generator(batch) if self.generator else None
        
        # If VQA module exists: get prediction
        predict = self.predictor(batch) if self.predictor else None
        
        return predict, caption, batch['v_att']

    def forward_vqa(self, batch):
        self.gradients = []
        batch = self.encoder(batch)
        predict = self.predictor(batch) if self.predictor else None
        return predict

    def forward_cap(self, batch):
        self.gradients = []
        batch = self.encoder(batch)
        caption = self.generator(batch) if self.generator else None
        return caption

def set_model(  encoder_type: str = 'base',
                predictor_type: str = 'base',
                decoder_type: str = 'base',
                ntoken: int = 0,
                v_dim: int = 0,
                embed_dim: int = 0,
                hidden_dim: int = 0,
                decoder_hidden_dim: int = 0,
                rnn_layer: int = 0,
                ans_dim: int = 0,
                cls_layer: int = 0,
                c_len: int = 0,
                device: str = '',
                dropout: float = 0.5,
                neg_slope: float = 0.5,
                rnn_type: str = 'GRU',
                att_type: str = 'base',
                conv_layer: int = 2,
                conv_type: str = 'corr',
                decoder_device: str = '',
                pretrained_embed_path: str = ''
):
    if decoder_device == '':
        print('set same as device')
        decoder_device = device
    return Wrapper( set_encoder(
                        encoder_type=encoder_type,
                        ntoken=ntoken,
                        v_dim=v_dim,
                        embed_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        device=device,
                        dropout=dropout,
                        rnn_type=rnn_type,
                        rnn_layer=rnn_layer,
                        att_type=att_type,
                        conv_type=conv_type,
                        conv_layer=conv_layer,
                        vocab_path=pretrained_embed_path
                    ).to(device),
                    set_predictor(
                        predictor_type=predictor_type,
                        v_dim=v_dim,
                        embed_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        ans_dim=ans_dim,
                        device=device,
                        cls_layer=cls_layer,
                        dropout=dropout,
                        c_len=c_len,
                        neg_slope=neg_slope
                    ).to(device) if predictor_type != 'none' else None,
                    set_decoder(
                        decoder_type=decoder_type,
                        ntoken=ntoken,
                        embed_dim=embed_dim,
                        hidden_dim=decoder_hidden_dim,
                        v_dim=v_dim,
                        max_len=c_len,
                        device=decoder_device,
                        dropout=dropout,
                        rnn_type=rnn_type,
                        att_type=att_type
                    ).to(decoder_device) if decoder_type != 'none' else None
            )