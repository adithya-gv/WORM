import torch
import torch.nn as nn
import copy

class EarlyBirdGradient():

    def __init__(self, chi=0.05):
        self.mask = None
        self.oldWeights = None
        self.chi = chi
    
    def updateMask(self, mask):
        self.mask = mask

    def clipGradients(self, model, device):
        if self.mask == None:
            return model
        index = 0
        self.mask = self.mask.to(device)
        for m in model.modules():
            m = m.to(device)
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                # If mask entry is 0, multiply gradients by 0.01
                m.weight.grad.data.mul_(self.mask[index:(index+size)] + self.chi * (1 - self.mask[index:(index+size)]))
                m.bias.grad.data.mul_(self.mask[index:(index+size)] + self.chi * (1 - self.mask[index:(index+size)]))

                index += size
        
        return model


                