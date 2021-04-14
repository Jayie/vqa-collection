import torch
import torch.nn as nn
from modules.encoder import set_encoder
from modules.predictor import set_predictor
from modules.generator import set_decoder
from modules.modules import PretrainedWordEmbedding

def use_pretrained_embedding(model, vocab_path: str, device: str):
    """ Replace the embedding layer in the model with the pre-trained embedding layer.
    Input:
        vocab_path: path for loading pre-trained word vectors
        device: device
    """
    model.embedding = PretrainedWordEmbedding(vocab_path=vocab_path, device=device)
    return model


class Wrapper(nn.Module):
    def __init__(self, device: str, encoder=None, predictor=None, generator=None):
        super().__init__()
        self.device = device
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
        # If encoder exists: get visual and text embeddings
        # Else: get original features
        if self.encoder != None:
            self.gradients = []
            batch = self.encoder(batch)
        else: batch = {'v': batch['img'].to(self.device)}

        # If Caption module exists: generate caption
        caption = self.generator(batch['v'], batch['c'], batch['cap_len'], batch['c_target']) if self.generator else None
        
        # If VQA module exists: get prediction
        predict = self.predictor(batch) if self.predictor else None
        
        return predict, caption, batch['v_att']

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
):
    return Wrapper( device, 
                    set_encoder(
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
                    ),
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
                    ),
                    set_decoder(
                        decoder_type=decoder_type,
                        ntoken=ntoken,
                        embed_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        v_dim=v_dim,
                        max_len=c_len,
                        device=device,
                        dropout=dropout,
                        rnn_type=rnn_type,
                        att_type=att_type
                    )
            )