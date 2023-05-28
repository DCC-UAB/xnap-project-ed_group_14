import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import csv
from tqdm import tqdm
def get_caps_from(features_tensors, model, vocab):
    #generate the caption
    model.eval()
    
    features = model.encoder(features_tensors.to(device))
    caps,alphas = model.decoder.generate_caption(features,vocab=vocab)
    caption = ' '.join(caps)
    #show_image(features_tensors[0],title=caption)
    
    return caption,alphas

def save_predictions(preds, cap):
    with open('results.csv', 'w', newline='') as file:
        escritor_csv = csv.writer(file)
        for r, p in zip(cap,preds):
            escritor_csv.writerow([r,p])
            

def predict(data_loader, model):
    #show any 1
    #dataiter = iter(data_loader)
    #images, caption = next(dataiter)
    preds=[]
    real_caption = []
    for images, cap in data_loader:
        
        l = [data_loader.dataset.vocab.itos[x.item()] for x in cap[0]]
        r = ' '.join(l)

        real_caption.append(r)
        
        print('REAL CAPTION:  ', r)
        img = images[0].detach().clone()
        #img1 = images[0].detach().clone()
        caps,alphas = get_caps_from(img.unsqueeze(0), model, data_loader.dataset.vocab)
        print('PREDICTED CAPTION: ', caps)
        preds.append(caps)
        #plot_attention(img, caps, alphas)
        print('\n')
    
    save_predictions(preds, real_caption)