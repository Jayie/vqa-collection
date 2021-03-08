import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch


def sample_one_batch(dataset, sample=0):
    """Directly transpose the sample data into a batch."""
    batch = {}
    batch['feature'] = dataset.questions[sample]['img_file']
    batch['feature_path'] = dataset.feature_path
    batch['q_word'] = dataset.questions[sample]['q_word']
    batch['target'] = dataset.answers[sample]
    data = dataset[sample]
    for i in data:
        if type(data[i]) == np.ndarray:
            shape = [1]
            shape.extend(data[i].shape)
            batch[i] = torch.from_numpy(data[i].reshape(shape))
        else:
            batch[i] = torch.Tensor([data[i]])
    return batch

def show_top_k_regions(model, dataset, ans_list, sample=0, img_path='../COCO', k=3):
    """Show the top-k relevant regions according to the questions."""
    # sample batch
    batch = sample_one_batch(dataset, sample)

    model.eval()
    # Get prediction and attention map
    predict, _, att = model(batch)
    att = att.squeeze()

    # Prepare image and bbox
    img_file = batch['feature'][:-3] + 'jpg'
    img = Image.open(os.path.join(img_path, os.path.basename(dataset.feature_path), img_file))
    bbox = np.load(os.path.join(batch['feature_path'], batch['feature']))['bbox']
    
    # Setup background (transparency=0.3)
    output = img.copy()
    output.putalpha(30)

    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()

    # Find top-k relevant regions
    value, index = att.topk(k)
    value = value.tolist()
    index = index.tolist()

    # Show the regions
    for i in range(1,k+1):
        b = bbox[index[-i]]
        region = img.crop([b[0], b[1], b[2], b[3]])
        if i != k:
            region.putalpha(128)
        output.paste(region, (int(b[0]), int(b[1])))

    # Draw rectangles and texts
    for i in range(1,k):
        b = bbox[index[i]]
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], fill=None, outline='red', width=2)
        text = f'{value[i]:.2f}'
        w, h = font.getsize(text)
        draw.rectangle([(b[0], b[1]), (b[0]+w+1, b[1]+h+1)], fill='red')
        draw.text([b[0], b[1]], text)
    
    # Print results
    print('Q:', batch['q_word'])
    print('\npredict: ', ans_list[torch.argmax(predict).item()])
    print('\ntarget:')
    for i, j in batch['target'].items():
        print(f'{min(j,3)/3:.2f}', ans_list[int(i)])
    return output