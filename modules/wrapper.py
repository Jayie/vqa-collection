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

    def forward(self, batch):
        # If encoder exists: get visual and text embeddings
        # Else: get original features
        v, w, _ = self.encoder(batch) if self.encoder!=None else (batch['img'].to(self.device), None, None)

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
        
        return predict, caption

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
):
    set_encoder = None
    set_predictor = None
    set_generator = None

    # TODO: initialize modules according to model_type

    if model_type == 'base':
        set_encoder = encoder.BaseEncoder(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, device, dropout, rnn_type)
        set_predictor = predictor.BasePredictor(v_dim, hidden_dim, ans_dim, device, cls_layer, dropout)
    
    elif model_type == 'new':
        set_encoder = encoder.NewEncoder(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, device, dropout, rnn_type)
        set_predictor = predictor.BasePredictor(v_dim, hidden_dim, ans_dim, device, cls_layer, dropout)
    
    elif model_type == 'cap':
        set_encoder = encoder.BaseEncoder(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, device, dropout, rnn_type)
        set_generator = CaptionDecoder(ntoken, embed_dim, hidden_dim, v_dim, c_len, device, dropout, rnn_type)

    elif model_type == 'vqa-e':
        set_encoder = encoder.BaseEncoder(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, device, dropout, rnn_type)
        set_predictor = predictor.BasePredictor(v_dim, hidden_dim, ans_dim, device, cls_layer, dropout)
        set_generator = CaptionDecoder(ntoken, embed_dim, hidden_dim, v_dim, c_len, device, dropout, rnn_type)
    
    elif model_type == 'q-cap':
        set_encoder = encoder.CaptionEncoder(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, c_len, device, dropout, rnn_type, neg_slope)
        set_predictor = predictor.PredictorwithCaption(v_dim, hidden_dim, ans_dim, device, cls_layer, dropout, neg_slope)
        set_generator = CaptionDecoder(ntoken, embed_dim, hidden_dim, v_dim, c_len, device, dropout, rnn_type)


    return Wrapper(device, set_encoder, set_predictor, set_generator)