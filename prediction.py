import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_caps_from(features_tensors, model, vocab):
    #generate the caption
    model.eval()
    
    features = model.encoder(features_tensors.to(device))
    caps,alphas = model.decoder.generate_caption(features,vocab=vocab)
    caption = ' '.join(caps)
    #show_image(features_tensors[0],title=caption)
    
    return caps,alphas



def predict(data_loader, model):
    #show any 1
    dataiter = iter(data_loader)
    images,caption = next(dataiter)
    print('REAL CAPTION:  ', caption[0])
    img = images[0].detach().clone()
    #img1 = images[0].detach().clone()
    caps,alphas = get_caps_from(img.unsqueeze(0), model, data_loader.dataset.vocab)
    print('PREDICTED CAPTION: ', caps)
    #plot_attention(img, caps, alphas)