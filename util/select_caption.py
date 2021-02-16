import torch
import torch.nn as nn
import numpy as np

def similarity(a, b):
    """Compute the similarity between string a and b.
    This is described in Ch.3.1 of "VQA-E: Explaining, Elaborating, and Enhancing Your Answers for Visual Questions"
    Input:
        a: [a_len, embed_dim]
        b: [b_len, embed_dim]
    Output: the similarity score between a and b
    """
    output = 0
    b_len = b.size(0)

    # For all word w_a in a, find the word w_b in b that is most related to w_a,
    # and then sum up similarity(w_a, w_b)
    for i in range(b_len):
        score = nn.functional.cosine_similarity(b, a[i,:].unsqueeze(0))
        score = torch.max(score).item()
        print(i, score)
        output += score
    return output / b_len

def similarity_given_caption(question, answer, caption):
    # TODO: compute the similarity between the caption and question-answer pair
    return

def select_caption():
    # TODO: select the relevant captions with the given Q-A pair.
    return