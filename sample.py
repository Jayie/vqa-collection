import json
import torch
from tqdm import tqdm

# TODO: need to rewrite

def sample_vqa(model, dataloader, ans_list, device, logger=None, sample=0):
    model = model.to(device)
    model.eval()
    count = [0] * len(ans_list)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i == sample and sample != 0: break
            
            # sample the first input for each batch
            index = batch['id'].tolist()[0]
            target = batch['a'].tolist()[0]
            predict = model(batch)
            predict = predict.argmax(1).tolist()
            answer  = predict[0]

            result = (str(index).zfill(12)
                + ' | '
                + 'Q: ' + dataloader.dataset.questions[index]['q_word']
                + '? | A: ' + ans_list[answer]
                + f' (score: {target[answer]:.2f})'
            )
            # print(result)

            if logger != None:
                logger.write(result)

            for j in predict:
                count[int(j)] += 1

    output = {}
    for i in range(len(ans_list)):
        if count[i] != 0: output[ans_list[i]] = count[i]

    return output