import torch

def sample_vqa(model, dataloader, ans_list, device, sample=10):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i == sample: return

            target = batch['a'].float().to(device).squeeze()
            predict = model(batch)
            predict = torch.max(predict, 1)[1].item()

            print(
                'Q: ' + dataloader.dataset.questions[i]['q_word']
                + '?\nA: ' + ans_list[predict]
                + f' (score: {target[predict].item():.2f})'
            )
            print()