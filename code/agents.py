import train
import torch

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

def earlyBirdRLAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, epochs, device):
    # Initialize environment
    env = Environment(model, device, trainloader, optimizer, criterion, testloader, earlyBird.ratio, 0.5, earlyBird, 0.9)
    env.init_training()