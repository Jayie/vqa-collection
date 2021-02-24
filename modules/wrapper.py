import torch.nn as nn
import modules

class Wrapper(nn.Module):
    def __init__(self, encoder=None, predictor=None, generator=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.generator = generator

    def forward(self, batch):
        # If encoder exists: get visual and text embeddings
        # Else: get original features
        v, w = self.encoder(batch) if self.encoder else (batch['img'], None)

        # If VQA module exists: get prediction
        predict = self.predictor(v, w) if self.predictor else None

        # If Caption module exists: generate caption
        caption = None
        if self.generator:
            caption = self.generator(v, batch['c'], batch['cap_len'])
        
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
                max_len: int = 0,
                device: str = '',
                dropout: float = 0.5,
                rnn_type: str = 'GRU',
):
    encoder = None
    predictor = None
    generator = None

    # TODO: initialize modules according to model_type

    # if model_type == 'base':
    #     encoder = models.encoder.BaseEncoder(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, device, dropout, rnn_type)
    #     predictor = models.predictor.BasePredictor(v_dim, hidden_dim, ans_dim, device, cls_layer, dropout)
    
    # elif model_type == 'new':
    #     encoder = models.encoder.NewEncoder(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, device, dropout, rnn_type)
    #     predictor = models.predictor.BasePredictor(v_dim, hidden_dim, ans_dim, device, cls_layer, dropout)
    
    # elif model_type == 'vqa-e':
    #     encoder = models.encoder.BaseEncoder(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, device, dropout, rnn_type)
    #     predictor = models.predictor.BasePredictor(v_dim, hidden_dim, ans_dim, device, cls_layer, dropout)
    #     generator = models.generator.CaptionDecoder(ntoken, embed_dim, hidden_dim, v_dim, max_len, device, dropout, rnn_type)
    
    # elif model_type == 'q-cap':
    #     encoder = models.encoder.BaseEncoder(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, device, dropout, rnn_type)
    #     # TODO: predictor
    #     generator = models.generator.CaptionDecoder(ntoken, embed_dim, hidden_dim, v_dim, max_len, device, dropout, rnn_type)


    return Wrapper(encoder, predictor, generator)