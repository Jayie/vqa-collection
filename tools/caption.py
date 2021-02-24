import numpy as np
import torch

def decode_with_beam_search(
    encoder,
    decoder,
    batch: dict,
    vocab: dict,
    device: str,
    k: int = 3
):
    """Generate captions with Beam Search.
    Input:
        encoder: encoder model
        decoder: decoder model
        batch: datas in one batch
        vocab: word dictionary
        device: device
        k: select top-k candidates (default = 3)
    """
    
    # Get visual features
    v, _ = encoder(batch) # [1, num_objs, v_dim]
    num_objs = v.size(1)
    v_dim = v.size(2)


    # Tensor to store top-k sequence (Initial: <start>)
    captions = torch.Tensor([[vocab['<start>']]]).to(device) # [k, 1]
    # Tensor to store scores of top-k sequence
    top_k_scores = torch.zeros(k, 1).to(device)
    # Tensor to store alphas of top-k sequence
    top_k_alphas = torch.zeros(k, 1, num_objs, v_dim).to(device) # [k, num_objs, v_dim]

    # Start decoding
    step = 1
    h1 = decoder.init_hidden(1)
    h2 = decoder.init_hidden(1)
    while True:
        scores, h1, h2, att = decoder.decode(v, captions, h1, h2)

        # Add
        # scores = top_k_scores.expand_as(scores) + scores
        # Error?

        # For the first step all k points will have the same scores (since same k previous word, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        # Convert unrolled indices to actual indices of scores
        # TODO