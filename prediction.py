import torch_directml
device = torch_directml.device()

from tqdm import tqdm
def get_caps_from(features_tensors, model, vocab):
    #generate the caption
    model.eval()
    
    features = model.encoder(features_tensors.to(device))
    caps,alphas = model.decoder.generate_caption(features,vocab=vocab)
    caption = ' '.join(caps)
    #show_image(features_tensors[0],title=caption)
    
    return caption,alphas



def predict(data_loader, model):
    #show any 1
    dataiter = iter(data_loader)
    images, caption = next(dataiter)

    for images, cap in tqdm(enumerate(data_loader)):
        
        l = [data_loader.dataset.vocab.itos[x.item()] for x in cap[0]]
        r = ' '.join(l)
        
        print('REAL CAPTION:  ', r)
        img = images[0].detach().clone()
        #img1 = images[0].detach().clone()
        caps,alphas = get_caps_from(img.unsqueeze(0), model, data_loader.dataset.vocab)
        print('PREDICTED CAPTION: ', caps)
        #plot_attention(img, caps, alphas)
        print('\n')