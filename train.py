from tqdm.auto import tqdm
import torch
import wandb
from model import save_model
#Only Raul
import torch_directml
device = torch_directml.device()


#def train(model, loader, criterion, optimizer, config):
#    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
#    wandb.watch(model, criterion, log="all", log_freq=10)
#
#    # Run training and track with wandb
#    total_batches = len(loader) * config.epochs
#    example_ct = 0  # number of examples seen
#    batch_ct = 0
#    for epoch in tqdm(range(config.epochs)):
#        for _, (images, labels) in enumerate(loader):
#
#            loss = train_batch(images, labels, model, optimizer, criterion)
#            example_ct +=  len(images)
#            batch_ct += 1
#
#            # Report metrics every 25th batch
#            if ((batch_ct + 1) % 25) == 0:
#                train_log(loss, example_ct, epoch)
#
#
#def train_batch(images, labels, model, optimizer, criterion, device="cuda"):
#    images, labels = images.to(device), labels.to(device)
#    
#    # Forward pass ➡
#    outputs = model(images)
#    loss = criterion(outputs, labels)
#    
#    # Backward pass ⬅
#    optimizer.zero_grad()
#    loss.backward()
#
#    # Step with optimizer
#    optimizer.step()
#
#    return loss


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


def train(model, optimizer, criterion, epochs, data_loader_train, vocab, data_loader_test):
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
                
        
        model.eval()
        print('STARTING TEST\n')
        
        for idx, (image, captions) in tqdm(enumerate(iter(data_loader_test))):
            image, captions = image.to(device), captions.to(device)
            outputs, attentions = model(image, captions)
            targets = captions[:, 1:]
            targets = targets.cpu()
            outputs = outputs.cpu()
            loss = criterion(outputs.view(-1, len(data_loader_train.dataset.vocab)), targets.reshape(-1))
            #print(loss.item())
            loss_test += loss.item()
            #print(loss_epoch)
                


        #save the latest model
        
        print(f'Epoch {epoch} Finished - Registering in WandB')
        #wandb.log({"total_loss": loss_epoch})
        wandb.log({'epoch':epoch, 'loss_train': loss_epoch/len(data_loader_train.dataset)})
        wandb.log({'epoch':epoch, 'loss_test': loss_test/len(data_loader_test.dataset)})
    

    print('EPOCHS FINISHED')
