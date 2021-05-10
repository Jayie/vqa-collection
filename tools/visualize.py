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
    img_id = str(int(dataset.questions[sample]['img_file'][-16:-4]))
    batch['c_word'] = dataset.captions[img_id]['c_word'][dataset.caption_id[sample]]
    data = dataset[sample]
    for i in data:
        if type(data[i]) == np.ndarray:
            shape = [1]
            shape.extend(data[i].shape)
            batch[i] = torch.from_numpy(data[i].reshape(shape))
        else:
            batch[i] = torch.tensor([data[i]])
    return batch


def show_att(att, img, bbox, k=3, output=None):
    """Given the attention scores, show the top-k relevant regions."""
    # Find top-k relevant regions
    value, index = att.topk(k)
    value = value.tolist()
    index = index.tolist()
    
    if output is None:
        output = img.copy()
        output.putalpha(30)

    # Show the regions
    for i in range(1,1+k):
        b = bbox[index[-i]]
        region = img.crop([b[0], b[1], b[2], b[3]])
        if value[-i] < max(value):
            region.putalpha(128)
        output.paste(region, (int(b[0]), int(b[1])))

    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    
    # Draw rectangles and texts
    for i in range(k):
        b = bbox[index[i]]
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], fill=None, outline='red', width=2)
        text = f'{value[i]:.2f}'
        w, h = font.getsize(text)
        draw.rectangle([(b[0], b[1]), (b[0]+w+1, b[1]+h+1)], fill='red')
        draw.text([b[0], b[1]], text)
    return output


def show_graph_att(model, dataset, ans_list, sample=0, img_path='../COCO', k=3, layer=-1):
    """Given a question, show the most important region and the top-k relevant regions according to the correlation scores."""
    # sample batch
    batch = sample_one_batch(dataset, sample)
    
    # Get prediction and graph attentions
    model.eval()
    with torch.no_grad():
        predict, att = model.get_att(batch)
        index = att.argmax().item()
        g_att = model.encoder(batch, True)[layer][0, index, :]
        g_att[index] = 1
    
    # Prepare image and bbox
    img_file = batch['feature'][:-3] + 'jpg'
    img = Image.open(os.path.join(img_path, os.path.basename(dataset.feature_path), img_file))
    bbox = np.load(os.path.join(batch['feature_path'], batch['feature']))['bbox']
    
    # Show top-k relevant regions
    output = show_att(g_att, img, bbox, k=k+1)
    
    # Print results
    print('Q:', batch['q_word'])
    print('C:', batch['c_word'])
    print('target:')
    for i, j in batch['target'].items():
        print(f'{min(j,3)/3:.2f}', ans_list[int(i)])
    print('\npredict: ', ans_list[torch.argmax(predict).item()])
    return output


def show_top_k_regions(model, dataset, ans_list, sample=0, img_path='../COCO', k=3):
    """Given a question, show the top-k relevant regions."""
    # sample batch
    batch = sample_one_batch(dataset, sample)

    # Get prediction and attention map
    model.eval()
    with torch.no_grad():
        predict, att = model.get_att(batch)
        att = att.squeeze()

    # Prepare image and bbox
    img_file = batch['feature'][:-3] + 'jpg'
    img = Image.open(os.path.join(img_path, os.path.basename(dataset.feature_path), img_file))
    bbox = np.load(os.path.join(batch['feature_path'], batch['feature']))['bbox']

    # Show the top-k relevant regions
    output = show_att(att, img, bbox, k=k)
    
    # Print results
    print('Q:', batch['q_word'])
    print('\npredict: ', ans_list[torch.argmax(predict).item()])
    print('\ntarget:')
    for i, j in batch['target'].items():
        print(f'{min(j,3)/3:.2f}', ans_list[int(i)])
    return output