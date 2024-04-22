import train
import torch
from earlyBirdGradient import EarlyBirdGradient

from environment import Environment

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

def fasterEarlyBirdAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, epochs, device, chi):
    ebg = EarlyBirdGradient(chi=chi)
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

def aggressiveClipAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, epochs, device):
    ebg = EarlyBirdGradient(chi=1)
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

def wormSTAR(model, criterion, optimizer, trainloader, testloader, earlyBird, epochs, device):
    ebg = EarlyBirdGradient(chi=1)
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

def ACearlyBERTAgent(model, train_dataset, tokenizer, eval_dataset, compute_metrics, epochs, earlyBird):
    ebg = EarlyBirdGradient(chi=1)
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

def ACearlyGemmaAgent(model, train_dataset, tokenizer, eval_dataset, compute_metrics, epochs, earlyBird):
    ebg = EarlyBirdGradient(chi=1)
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