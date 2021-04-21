import numpy as np
import torch

def decode_with_beam_search(
    encoder,
    decoder,
    batch: dict,
    vocab: dict,
    device: str,
    c_len: int = 20,
    k: int = 3
):
    """Generate captions with Beam Search. (Reference: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
    Input:
        encoder: encoder model
        decoder: decoder model
        batch: datas in one batch
        vocab: word dictionary
        device: device
        c_len: max length of captions
        k: select top-k candidates (default = 3)
    Output:
        generated_caption
        alpha
    """
    
    # Get visual features
    v = encoder(batch)['v'] # [1, num_objs, v_dim]
    num_objs = v.size(1)
    v_dim = v.size(2)

    # Initialize
    # Tensor to store top-k previous words (Initial: <start>)
    prev_words = torch.Tensor([[vocab['<start>']] for _ in range(k)], device=self.device) # [k, 1]
    # Tensor to store top-k sequences (Initial: <start>)
    seqs = prev_words
    # Tensor to store scores of top-k sequence
    top_k_scores = torch.zeros(k, 1, device=self.device) # [k, 1]
    # Tensor to store alphas of top-k sequence
    top_k_alphas = torch.ones(k, 1, num_objs, v_dim, device=self.device) # [k, num_objs, v_dim]

    # Lists to store completed sequences
    complete_seqs = list()
    complete_alphas = list()
    complete_scores = list()

    # Start decoding
    step = 1
    h1 = decoder.init_hidden(1)
    h2 = decoder.init_hidden(1)
    while True:
        scores, h1, h2, alpha = decoder.decode(v, prev_words, h1, h2)

        # Add
        # scores = top_k_scores.expand_as(scores) + scores
        # Error?

        # For the first step, all k points will have the same scores
        # (since same k previous word, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        ###########################################################################
        # TODO: Check result
        ###########################################################################

        # Convert unrolled indices to actual indices of scores
        prev_word = top_k_words / len(vocab)
        next_word = top_k_words % len(vocab)

        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word], next_word.unsqueeze(1)], dim=1) # [num_seq, step+1]
        top_k_alphas = torch.cat([top_k_alphas[prev_word], alpha[prev_word].unsqueeze(1)], dim=1) # [num_seq, step+1, num_objs]

        # Find incomplete sequences (didn't reach <end>)
        incomplete_inds = [ind for ind, next_word in enumerate(next_word) if next_word != vocab['<end>']]
        complete_inds = list(set(range(len(next_word))) - set(incomplete_inds))

        # Set aside complete sequence
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_scores.extend(top_k_scores[complete_inds].tolist())
            complete_alphas.extend(top_k_alphas[complete_inds].tolist())
        
        k -= len(complete_inds) # Reduce Beam length accordingly
        if k == 0: break # If all sequences reach <end>: end of while loop

        # Proceed with incomplete sequences
        h1 = h1[prev_word[incomplete_inds]]
        h2 = h2[prev_word[incomplete_inds]]
        seqs = seqs[incomplete_inds]
        top_k_alphas = top_k_alphas[incomplete_inds]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        prev_words = next_word[incomplete_inds].unsqueeze(1)

        # Break if length out of range
        if step > c_len: break
        step += 1

    # Return the sequence with the highest score
    i = complete_scores.index(max(complete_scores))
    return complete_seqs[i], complete_alphas[i]