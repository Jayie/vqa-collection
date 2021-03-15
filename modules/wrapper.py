import torch.nn as nn
import modules.encoder as encoder
import modules.predictor as predictor
from modules.generator import CaptionDecoder
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
        self.gradients = []

    def save_grad(self, grad):
        self.gradients.append(grad)

    def forward(self, batch):
        # If encoder exists: get visual and text embeddings
        # Else: get original features
        if self.encoder != None:
            v, w, att = self.encoder(batch)
            self.gradients = []
            v.register_hook(self.save_grad)
        else:
            v, w, att = (batch['img'].to(self.device), None, None)

        # If VQA module exists: get prediction
        predict = self.predictor(v, w) if self.predictor!=None else None
        del w

        # If Caption module exists: generate caption
        caption = None
        if self.generator:
            c = batch['c'].to(self.device)
            cap_len = batch['cap_len'].to(self.device)
            caption = self.generator(v, c, cap_len)

            c.detach()
            cap_len.detach()
            del c
            del cap_len
        
        return predict, caption, att

def set_model(  model_type: str,
                ntoken: int = 0,
                v_dim: int = 0,
                embed_dim: int = 0,
                hidden_dim: int = 0,
                rnn_layer: int = 0,
                att_fc_dim: int = 0,
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
    def set_encoder():
        if model_type == 'conv':
            return encoder.RelationEncoder(
                ntoken=ntoken,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                v_dim=v_dim,
                att_fc_dim=att_fc_dim,
                device=device,
                dropout=dropout,
                rnn_type=rnn_type,
                rnn_layer=rnn_layer,
                att_type=att_type,
                conv_type=conv_type,
                conv_layer=conv_layer
            )
        if model_type == 'q-cap':
            return encoder.CaptionEncoder(
                ntoken=ntoken,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                rnn_layer=rnn_layer,
                v_dim=v_dim,
                att_fc_dim=att_fc_dim,
                c_len=c_len,
                device=device,
                dropout=dropout,
                rnn_type=rnn_type,
                neg_slope=neg_slope
            )
        return encoder.BaseEncoder(
            ntoken=ntoken,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            v_dim=v_dim,
            att_fc_dim=att_fc_dim,
            device=device,
            dropout=dropout,
            rnn_type=rnn_type,
            rnn_layer=rnn_layer,
            att_type=att_type
        )

    def set_predictor():
        if model_type == 'cap': return None
        if model_type == 'q-cap': return predictor.PredictorwithCaption(
            v_dim=v_dim,
            hidden_dim=hidden_dim,
            ans_dim=ans_dim,
            device=device,
            cls_layer=cls_layer,
            dropout=dropout,
            neg_slope=neg_slope
        )
        return predictor.BasePredictor(
            v_dim=v_dim,
            hidden_dim=hidden_dim,
            ans_dim=ans_dim,
            device=device,
            cls_layer=cls_layer,
            dropout=dropout
        )

    def set_generator():
        if model_type in ['base', 'conv']: return None
        return CaptionDecoder(
            ntoken=ntoken,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            v_dim=v_dim,
            max_len=c_len,
            device=device,
            dropout=dropout,
            rnn_type=rnn_type,
        )

    return Wrapper(device, set_encoder(), set_predictor(), set_generator())