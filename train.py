from tqdm.auto import tqdm
import torch
import wandb
from model import save_model
# Metrics
from torchtext.data.metrics import bleu_score
from torchmetrics.text import Perplexity


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


def generate_predictions_sentences(model, image, targets, vocab):
    predictions = []
    # Calculate batch predictions
    model.eval()
    with torch.no_grad():
        img = image.detach().clone()
        features = model.encoder(img.to(device))
        for f in features:
            caps,_  = model.decoder.generate_caption(f.unsqueeze(0), vocab=vocab)
            predictions.append(caps)
    model.train()
    
    # Process real captions
    real_captions = []
    for t in targets:
        real_cap = [vocab.itos[idx] if ('<' not in vocab.itos[idx] and vocab.itos[idx] != '<UNK>') or vocab.itos[idx] == '.' else '' for idx in t.cpu().numpy()]
        real_caption = [real_cap[:real_cap.index('')]]
        real_captions.append(real_caption)
    return predictions, real_captions

def train(model, optimizer, criterion, epochs, data_loader_train, vocab, data_loader_test):
    
    perp_score = Perplexity().to(device)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_every = 100
    wandb.watch(model, criterion, log='all', log_freq=10)
    for epoch in tqdm(range(1,epochs+1)): 
        loss_epoch = 0  
        loss_test = 0
        print('STARTING TRAIN\n')
        model.train()
        for idx, (image, captions) in tqdm(enumerate(iter(data_loader_train))):
            image,captions = image.to(device),captions.to(device)
            # Zero the gradients.
            optimizer.zero_grad()
            # Feed forward
            outputs,attentions = model(image, captions)
            # Calculate the batch loss.
            targets = captions[:,1:]
            loss = criterion(outputs.view(-1, len(vocab)), targets.reshape(-1))
            # Backward pass.
            loss.backward()
            loss_epoch += loss.item()

            # Update the parameters in the optimizer.
            optimizer.step()
            if (idx+1)%print_every == 0:
                print(f'Epoch{epoch}:  Loss{loss.item()}')
                #wandb.log({'epoch':epoch, 'loss_batch_train': loss_epoch/len(data_loader_train.dataset)})
        
        # TRAIN METRICS
        dataiter = iter(data_loader_train)
        image, captions = next(dataiter)
        image, captions = image.to(device), captions.to(device)
        
        targets = captions[:, 1:]
        predictions, captions = generate_predictions_sentences(model, image, targets, data_loader_train.dataset.vocab)

        # BLEU
        bleu_train = bleu_score(predictions, captions)
        # Perplecxity
        shortest_sentence = min(targets.shape[1], outputs.shape[1])
        shortest_batch = min(targets.shape[0], outputs.shape[0])
        perp_train = perp_score(outputs[:shortest_batch, :shortest_sentence, :], targets[:shortest_batch, :shortest_sentence]).item()

        
        model.eval()
        print('STARTING TEST\n')
        with torch.no_grad():
            for idx, (image, captions) in tqdm(enumerate(iter(data_loader_test))):
                image, captions = image.to(device), captions.to(device)
                outputs, attentions = model(image, captions)
                targets = captions[:, 1:]
                #targets = targets.cpu()
                #outputs = outputs.cpu()
                loss = criterion(outputs.view(-1, len(data_loader_train.dataset.vocab)), targets.reshape(-1))
                #print(loss.item())
                loss_test += loss.item()
                #print(loss_epoch)
            
            # Metrics
            dataiter = iter(data_loader_test)
            image, captions = next(dataiter)
            image, captions = image.to(device), captions.to(device)

            targets = captions[:, 1:]
            predictions, captions = generate_predictions_sentences(model, image, targets, data_loader_train.dataset.vocab)

            # BLEU
            bleu_test = bleu_score(predictions, captions)

            # Perplexity
            shortest_sentence = min(targets.shape[1], outputs.shape[1])
            shortest_batch = min(targets.shape[0], outputs.shape[0])
            perp_test = perp_score(outputs[:shortest_batch, :shortest_sentence, :], targets[:shortest_batch, :shortest_sentence]).item()
                


        #save the latest model
        
        print(f'Epoch {epoch} Finished - Registering in WandB')
        #wandb.log({"total_loss": loss_epoch})
        wandb.log({'epoch':epoch, 'loss_train': loss_epoch/len(data_loader_train)})
        wandb.log({'epoch':epoch, 'loss_test': loss_test/len(data_loader_test)})
        
        wandb.log({'train_bleu': bleu_train / (len(data_loader_train))}) 
        wandb.log({'test_bleu': bleu_test / (len(data_loader_test))}) 
        wandb.log({'perp_train':perp_train})
        wandb.log({'perp_test':perp_test})
   

    print('EPOCHS FINISHED')
