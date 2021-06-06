import numpy as np
import torch
from tqdm import tqdm
from queue import PriorityQueue


from queue import PriorityQueue
class BeamSearchNode(object):
    """Data structure for Beam Search"""
    def __init__(self, h, prev, word, logp, length):
        """
        h: hidden state
        prev: pointer of previous node
        word: word id
        logp: log probability
        length: length of sequence
        """
        self.h = h
        self.prev = prev
        self.word = word
        self.logp = logp
        self.length = length
        
    def eval(self, alpha=1.0):
        reward = 0 # for shaping a reward
        return -(self.logp / float(self.length - 1 + 1e-6) + alpha * reward)
    
    def __repr__(self):
        return f'BSNode(l={self.length}: word={self.word}, logp={self.logp})'
    
    def __eq__(self, other):
        if other is None: return False
        return self.eval() == other.eval()
    
    def __lt__(self, other):
        if other is None: return True
        return self.eval() < other.eval()


def decode_with_beam_search(
    model,
    batch: dict,
    vocab_list: list,
    c_len: int = 20,
    k: int = 3,
    time_limit: int = 2000
):
    """
    reference: https://github.com/budzianowski/PyTorch-Beam-Search-Decoding
    """
    batch_size = batch['id'].size(0)
    device = model.generator.device
    decoded_batch = []

    model.eval()

    with torch.no_grad():
        embed = model.encoder(batch)

        for i in range(batch_size):
            # Initialize
            v = embed['v'][i].unsqueeze(0)
            v_mean = v.mean(1).to(device)
            h = model.generator.init_hidden(1)
            start = torch.LongTensor([[vocab_list.index('<start>')]]).to(device)

            # Number of sentences to generate
            endnodes = []

            # Initialize beam search queue
            nodes = PriorityQueue()
            node = BeamSearchNode(h=h, prev=None, word=start, logp=0, length=1)
            nodes.put((node.eval(), node))
            q_size = 1

            while True:
                # Give up when decoding takes too long
                if q_size > 2000: break

                # fetch the best word
                score, prev = nodes.get()

                # Record if reach the end of sentence
                if (prev.word == vocab_list.index('<end>') or prev.length >= c_len) and prev.prev != None:
                    endnodes.append((score, prev))
                    # Break when we get enough generated sentences
                    if len(endnodes) >= k: break
                    else: continue

                # Get word embedding and hidden state of previous word
                word = torch.LongTensor([[prev.word]]).to(device)
                h = prev.h.to(device)
                encode = model.encoder.embedding(word).squeeze(1)

                # Decode
                h, distribution = model.generator.decode(v=v, prev=encode, v_mean=v_mean, h=h)

                # Get top-k prediction
                prob, word_id = distribution.topk(k=k, largest=True, sorted=True)

                # Put into queue
                for j in range(k):
                    node = BeamSearchNode(
                        h=h,
                        prev=prev,
                        word=word_id[0,j],
                        logp=prob[0,j].item() + prev.logp,
                        length=prev.length+1
                    )
                    nodes.put((node.eval(), node))

                q_size += k - 1

            # Chose k-best paths
            if len(endnodes) < k:
                l = k-len(endnodes)
                endnodes = [nodes.get() for _ in range(l)]

            output = []
            for score, node in sorted(endnodes, key=lambda x: -x[0]):
                seq = []
                while node is not None:
                    seq.append(vocab_list[node.word])
                    node = node.prev
                output.append((score, ' '.join(seq[::-1])))
            decoded_batch.append(output.copy())

    return decoded_batch