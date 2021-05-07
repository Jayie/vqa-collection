import torch
import torch.nn as nn
from modules.encoder import set_encoder
from modules.predictor import set_predictor
from modules.generator import set_decoder


def compute_score(predict, target, device, get_label=False):
    """Compute score (according to the VQA evaluation metric)"""
    predict = predict.to(device)
    target = target.to(device)

    # get the most possible predicted results for each question
    logits = torch.max(predict, 1)[1].data

    # transfer predicted results into one-hot encoding
    one_hots = torch.zeros(*target.size()).to(device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)

    scores = one_hots * target
    if get_label: return scores, logits
    return scores


def instance_bce_with_logits(predict, target):
    """Loss function for VQA prediction"""
    loss = nn.functional.binary_cross_entropy_with_logits(predict, target)
    loss *= target.size(1)
    return loss


def ce_for_language_model(predict, target):
    """ Loss function for caption generation"""
    assert predict.dim() == 2
    loss = nn.functional.cross_entropy(predict, target)
    return loss


class Wrapper(nn.Module):
    def __init__(self, encoder=None, predictor=None, generator=None, use_mtl=True):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.generator = generator
        self.device = encoder.device

        # Multi-task loss weight
        # Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics, CVPR 2018
        # Pytorch implementation reference: https://github.com/Hui-Li/multi-task-learning-example-PyTorch
        if use_mtl: self.log_vars = nn.Parameter(torch.zeros(2))
        self.use_mtl = use_mtl

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

    def get_loss(self, batch):
        predict, caption, _ = self.forward(batch)
        loss = torch.tensor(0, dtype=torch.float).to(self.device)
        write = {}

        if predict is not None:
            target = batch['a'].float().to(self.device)
            loss_vqa = instance_bce_with_logits(predict, target)
            if self.use_mtl:
                precision = torch.exp(-self.log_vars[0])
                loss += torch.sum(precision * loss_vqa + self.log_vars[0])
            else: loss += loss_vqa
            write['train/loss'] = loss_vqa.item()
            write['train/score'] = compute_score(predict, target, self.device).sum().item()

        if caption is not None:
            loss_cap = ce_for_language_model(caption['predict'], caption['target'])
            if self.use_mtl:
                precision = torch.exp(-self.log_vars[1])
                loss += torch.sum(precision * loss_cap + self.log_vars[1])
            else: loss += loss_cap
            write['train/cap/loss'] = loss_cap.item()
        
        loss = torch.mean(loss)
        return loss, write, self.log_vars.data.tolist()


    def forward_vqa(self, batch):
        self.gradients = []
        target = batch['a'].float().to(self.device)
        batch = self.encoder(batch)
        predict = self.predictor(batch)
        score, label = compute_score(predict, target, self.device, True)
        return score, label, target

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
                pretrained_embed_path: str = '',
                use_mtl: bool = False,
):
    if decoder_device == '':
        print('use same device')
        decoder_device = device
    return Wrapper( encoder = set_encoder(
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
                        device=device,
                        vocab_path=pretrained_embed_path,
                    ),
                    predictor = set_predictor(
                        predictor_type=predictor_type,
                        v_dim=v_dim,
                        embed_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        ans_dim=ans_dim,
                        device=device,
                        cls_layer=cls_layer,
                        dropout=dropout,
                        c_len=c_len,
                        neg_slope=neg_slope,
                        device=device
                    )
                    generator = set_decoder(
                        decoder_type=decoder_type,
                        ntoken=ntoken,
                        embed_dim=embed_dim,
                        hidden_dim=decoder_hidden_dim,
                        v_dim=v_dim,
                        max_len=c_len,
                        device=decoder_device,
                        dropout=dropout,
                        rnn_type=rnn_type,
                        att_type=att_type,
                        device=device,
                    ).to(decoder_device) if decoder_type != 'none' else None,
                    use_mtl = use_mtl
            )