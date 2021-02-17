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
    a_len = a.size(0)

    # For all word w_a in a, find the word w_b in b that is most related to w_a,
    # and then sum up similarity(w_a, w_b)
    for i in range(a_len):
        score = nn.functional.cosine_similarity(b, a[i,:].unsqueeze(0))
        score = torch.max(score).item()
        print(i, score)
        output += score
    return output / a_len


def select_caption(question, answer, captions):
    """Compute the similarity between the caption and question-answer pair.
    Input:
        question: [q_len, embed_dim]
        answer: [a_len, embed_dim]
        caption: [num_captions, c_len, embed_dim]
    """
    get_similarity = lambda q, a, c: (similarity(q, c) + similarity(a, c)) / 2
    best_score = 0
    best_index = 0
    for i in range(len(captions)):
        temp = get_similarity(question, answer, captions)
        if temp > best_score:
            best_index = i
            best_score = temp
    # TODO: Given a question-answer pair, select the most relevant caption.
    return best_index