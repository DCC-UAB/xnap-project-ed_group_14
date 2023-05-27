import torch_directml
device = torch_directml.device()

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
    for images, caption in data_loader:
        print('REAL CAPTION:  ', caption[0])
        img = images[0].detach().clone()
        #img1 = images[0].detach().clone()
        caps,alphas = get_caps_from(img.unsqueeze(0), model, data_loader.dataset.vocab)
        print('PREDICTED CAPTION: ', caps)
        #plot_attention(img, caps, alphas)
        print('\n\n')