import wandb
import torch

import torch_directml
device = torch_directml.device()
#
#def test(model, test_loader, device="cuda", save:bool= True):
#    # Run the model on some test examples
#    with torch.no_grad():
#        correct, total = 0, 0
#        for images, labels in test_loader:
#            images, labels = images.to(device), labels.to(device)
#            outputs = model(images)
#            _, predicted = torch.max(outputs.data, 1)
#            total += labels.size(0)
#            correct += (predicted == labels).sum().item()
#
#        print(f"Accuracy of the model on the {total} " +
#              f"test images: {correct / total:%}")
#        
#        wandb.log({"test_accuracy": correct / total})
#
#    if save:
#        print(len(images))
#        # Save the model in the exchangeable ONNX format
#        torch.onnx.export(model,  # model being run
#                          images,  # model input (or a tuple for multiple inputs)
#                          "model.onnx",  # where to save the model (can be a file or file-like object)
#                          export_params=True,  # store the trained parameter weights inside the model file
#                          opset_version=10,  # the ONNX version to export the model to
#                          do_constant_folding=True,  # whether to execute constant folding for optimization
#                          input_names=['input'],  # the model's input names
#                          output_names=['output'],  # the model's output names
#                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
#                                        'output': {0: 'batch_size'}})
#        wandb.save("model.onnx")


def test(model, data_loader, vocab_size, criterion):
    realcaption_list = []
    caption_lists = []
    images_list = []

    model.eval()
    with torch.no_grad():
        for images, realcaptions in data_loader:
            images, realcaptions = images.to(device),realcaptions.to(device)

            #vocab = list(dataset.vocab.itos.values())
            #realcaption_letter = [[vocab[int(idx)] for idx in array] for array in realcaptions.cpu().numpy()]
            #realcaption_list.extend(realcaption_letter)

            outputs, attentions = model(images, realcaptions)
            targets = realcaptions[:, 1:]
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))



            #features = model.encoder(images)
            #for feature, img in zip(features, images):
            #    feature = feature.unsqueeze(0)
            #    caps, alphas = model.decoder.generate_caption(feature, v_c)
            #    # caption = ' '.join(caps)
            #    caption_lists.append(caps)
            #    images_list.append(img.detach().cpu())

    return realcaption_list, caption_lists, images_list
