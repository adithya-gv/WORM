import torch
import torch.nn as nn
import train
import numpy as np

from earlyBird import EarlyBird

class Environment():

    ACTION_TABLE = {0, 1}

    def __init__(self, model, device, trainloader, optimizer, criterion, testloader, ratio, alpha, earlyBird, gamma):
        self.device = device
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.testloader = testloader
        self.ratio = ratio
        self.alpha = alpha
        self.earlyBird = earlyBird
        self.gamma = gamma

        # Save initial model weights
        self.initial_weights = model.state_dict()

        # Initialize Q-Table
        self.Q_table = {}
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # compute reward on the action of pruning and continuing training
    def compute_reward_proceed(self, model, epoch):
        # Apply Mask to Model
        mask = self.earlyBird.pruning(model, self.ratio)
        new_model = self.earlyBird.apply_mask(model, mask)

        # Compute Mask Distance
        mask_distance = self.earlyBird.compute_mask_distance(mask)

        # Determine if EarlyBird traditional approach would have converged
        early_bird_converged = self.earlyBird.early_bird_emerge(model)

        penalty = 0
        # If it decided to converge, apply major penalty.
        if early_bird_converged:
            penalty = 100

        # Train model for one epoch
        train.train_one_epoch(new_model, self.device, self.trainloader, self.optimizer, self.criterion, epoch)

        # Test model to retrieve accuracy
        acc = train.test(new_model, self.testloader, self.device)

        # Compute Reward
        reward = acc * 10 * self.sigmoid(1 / mask_distance) - ((epoch - 3) / 10) - penalty

        return reward

    # compute reward on the action of pruning and stopping training
    def compute_reward_terminate(self, model, epoch):
        # Apply mask to model
        mask = self.earlyBird.pruning(model, self.ratio)
        new_model = self.earlyBird.apply_mask(model, mask)

        # Compute mask distance
        mask_distance = self.earlyBird.compute_mask_distance(mask)

        # Train model for one epoch
        train.train_one_epoch(new_model, self.device, self.trainloader, self.optimizer, self.criterion, epoch)

        # Test old model to retrieve accuracy
        target_acc = train.test(model, self.testloader, self.device)

        # Test model to retrieve accuracy
        acc = train.test(new_model, self.testloader, self.device)

        # if acc is less than target accuracy within range, apply major penalty
        penalty = 0
        if acc < target_acc - 2:
            penalty = 100

        # Compute Reward
        reward = acc * 10 * self.sigmoid(1 / mask_distance) - (epoch / 10) - penalty
        
        return reward

    def take_action(self, model, epoch):
        # Randomly select an action
        action = np.random.choice(list(self.ACTION_TABLE), p = [max(1 - (5 * epoch / 10), 0.3), max(0 + (5 * epoch / 10), 0.7)])

        reward = 0
        # Prune
        if action == 0:
            reward = self.compute_reward_proceed(model, epoch)
        
        # No Prune
        else:
            reward = self.compute_reward_terminate(model, epoch)

        # Update Q-table
        temporal_difference = max(self.Q_table[(epoch + 1, a)] for a in self.ACTION_TABLE) - self.Q_table[(epoch, action)]
        self.Q_table[(epoch, action)] = (1 - self.alpha) * self.Q_table[(epoch, action)] + self.alpha * (reward + self.gamma * temporal_difference)

        return action
    
    def init_training(self):
        # Populate Q-Table for 100 epochs on both actions, with a value of -1
        for epoch in range(100):
            for action in self.ACTION_TABLE:
                self.Q_table[(epoch, action)] = -1
    
    def restart_training(self, model):
        # Reset model weights
        model.load_state_dict(self.initial_weights)

        # Reset optimizer parameters
        self.optimizer.params = model.parameters()

        return 0
    

    def inference(self, epoch):
        # Assuming that we have a trained q_table, select and return the ideal action at each epoch step
        return max(self.Q_table[(epoch, action)] for action in self.ACTION_TABLE)

