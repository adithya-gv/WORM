import train
import torch
import torch.nn as nn
from earlyBird import EarlyBirdGradient
from transformers.models import bert, gemma

def worm_hook(ebg):
    def gradient_hook(grad):
        mask = ebg.getMask()
        chi = ebg.getChi()
        if mask is None:
            return grad
        return grad * (mask + chi * (1 - mask))
    return gradient_hook

# Build the early-bird agent as described in the paper.
def earlyBirdAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, epochs, device):

    for epoch in range(epochs):
        if earlyBird.early_bird_emerge(model):
            print("Early Bird Found!")

            # Save model
            torch.save(model.state_dict(), "early_bird_model.pth")

            # Test Model
            train.test(model, testloader, device)

            # Output epoch number
            print(f"Epoch: {epoch}")
            break

        # Otherwise, Train the model
        train.train_one_epoch(model, device, trainloader, optimizer, criterion, epoch)

        # Test Model
        train.test(model, testloader, device)

def gradClipEBAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, epochs, device, chi):
    ebg = EarlyBirdGradient(chi=chi)
    for _, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for _, param in module.named_parameters():
                if param.requires_grad:
                    param.register_hook(worm_hook(ebg))
    for epoch in range(epochs):
        if earlyBird.early_bird_emerge(model):
            print("Early Bird Found!")

            # Save model
            torch.save(model.state_dict(), "early_bird_model.pth")

            # Test Model
            train.test(model, testloader, device)

            # Output epoch number
            print(f"Epoch: {epoch}")
            break
        else:
            mask = earlyBird.masks[-1]
            ebg.updateMask(mask)

        # Otherwise, Train the model
        train.train_one_epoch_with_clip(model, device, trainloader, optimizer, criterion, epoch, ebg)

        # Test Model
        train.test(model, testloader, device)

def greedyClipEBAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, epochs, device):
    ebg = EarlyBirdGradient(chi=1)
    for _, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for _, param in module.named_parameters():
                if param.requires_grad:
                    param.register_hook(worm_hook(ebg))
    for epoch in range(epochs):
        if earlyBird.early_bird_emerge(model):
            print("Early Bird Found!")

            # Save model
            torch.save(model.state_dict(), "early_bird_model.pth")

            # Test Model
            train.test(model, testloader, device)

            # Output epoch number
            print(f"Epoch: {epoch}")
            break
        else:
            mask = earlyBird.masks[-1]
            ebg.updateMask(mask)
            dist = earlyBird.get_mask_distance()
            ebg.updateChi(dist)

        # Otherwise, Train the model
        train.train_one_epoch_with_clip(model, device, trainloader, optimizer, criterion, epoch, ebg)

        # Test Model
        train.test(model, testloader, device)

def wormEBAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, epochs, device):
    ebg = EarlyBirdGradient(chi=1)
    for _, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for _, param in module.named_parameters():
                if param.requires_grad:
                    param.register_hook(worm_hook(ebg))
    for epoch in range(epochs):
        if earlyBird.early_bird_emerge(model):
            print("Early Bird Found!")

            # Save model
            torch.save(model.state_dict(), "early_bird_model.pth")

            # Test Model
            train.test(model, testloader, device)

            # Output epoch number
            print(f"Epoch: {epoch}")
            break
        else:
            mask = earlyBird.masks[-1]
            ebg.updateMask(mask)
            dist = earlyBird.get_mask_distance()
            if (dist < 0.15):
                ebg.updateChi(0.003)

        # Otherwise, Train the model
        train.train_one_epoch_with_clip(model, device, trainloader, optimizer, criterion, epoch, ebg)

        # Test Model
        train.test(model, testloader, device)


def earlyBERTAgent(model, train_dataset, tokenizer, eval_dataset, compute_metrics, epochs, earlyBird):
    accuracy = 0
    for epoch in range(epochs):
        if earlyBird.early_bird_emerge(model):
            print("Early Bird Found!")

            # Test Model
            accuracy = train.test_transformer(model, eval_dataset, tokenizer, compute_metrics)

            # Output epoch number
            print(f"Epoch: {epoch}")
            break

        # Otherwise, Train the model
        train.train_one_epoch_transformer(model, train_dataset, tokenizer, compute_metrics)

        # Test Model
        train.test_transformer(model, eval_dataset, tokenizer, compute_metrics)
    
    return accuracy

def wormBERTAgent(model, train_dataset, tokenizer, eval_dataset, compute_metrics, epochs, earlyBird):
    ebg = EarlyBirdGradient(chi=1)
    for _, module in model.named_modules():
        if isinstance(module, bert.modeling_bert.BertSelfAttention):
            for _, param in module.named_parameters():
                if param.requires_grad:
                    param.register_hook(worm_hook(ebg))
    accuracy = 0
    for epoch in range(epochs):
        if earlyBird.early_bird_emerge(model) and (epoch >= earlyBird.epoch_keep):
            print("Early Bird Found!")

            # Test Model
            accuracy = train.test_transformer(model, eval_dataset, tokenizer, compute_metrics)

            # Output epoch number
            print(f"Epoch: {epoch}")
            break
        else:
            mask = earlyBird.masks[-1]
            ebg.updateMask(mask)
            dist = earlyBird.get_mask_distance()
            ebg.updateChi(dist)
            if (dist < 0.015):
                ebg.updateChi(0.01)

        # Otherwise, Train the model
        train.train_one_epoch_bert_transformer(model, train_dataset, tokenizer, compute_metrics, ebg)

        # Test Model
        train.test_transformer(model, eval_dataset, tokenizer, compute_metrics)
    
    return accuracy


def earlyGemmaAgent(model, train_dataset, tokenizer, eval_dataset, compute_metrics, epochs, earlyBird):
    accuracy = 0
    for epoch in range(epochs):
        if earlyBird.early_bird_emerge(model):
            print("Early Bird Found!")

            # Test Model
            accuracy = train.test_transformer(model, eval_dataset, tokenizer, compute_metrics)

            # Output epoch number
            print(f"Epoch: {epoch}")
            break

        # Otherwise, Train the model
        train.train_one_epoch_transformer(model, train_dataset, tokenizer, compute_metrics)

        # Test Model
        train.test_transformer(model, eval_dataset, tokenizer, compute_metrics)
    
    return accuracy

def wormGemmaAgent(model, train_dataset, tokenizer, eval_dataset, compute_metrics, epochs, earlyBird):
    ebg = EarlyBirdGradient(chi=1)
    for _, module in model.named_modules():
        if isinstance(module, gemma.modeling_gemma.GemmaSdpaAttention):
            for _, param in module.named_parameters():
                if param.requires_grad:
                    param.register_hook(worm_hook(ebg))
    accuracy = 0
    for epoch in range(epochs):
        if earlyBird.early_bird_emerge(model) and (epoch >= earlyBird.epoch_keep):
            print("Early Bird Found!")

            # Test Model
            accuracy = train.test_transformer(model, eval_dataset, tokenizer, compute_metrics)

            # Output epoch number
            print(f"Epoch: {epoch}")
            break
        else:
            mask = earlyBird.masks[-1]
            ebg.updateMask(mask)
            dist = earlyBird.get_mask_distance()
            ebg.updateChi(dist)
            if (dist < 0.015):
                ebg.updateChi(0.01)

        # Otherwise, Train the model
        train.train_one_epoch_transformer_clip(model, train_dataset, tokenizer, compute_metrics, ebg)

        # Test Model
        train.test_transformer(model, eval_dataset, tokenizer, compute_metrics)
    
    return accuracy
    