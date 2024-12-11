import torch
import torch.nn as nn
import copy
from transformers.models import bert

class EarlyBirdGradient():

    def __init__(self, chi=1):
        self.mask = None
        self.oldWeights = None
        self.chi = chi
        self.loss = 1e9
    
    def updateMask(self, mask):
        self.mask = mask

    def updateChi(self, dist):
        if (dist > 1):
            self.chi = 1
        else:
            self.chi = dist
    
    def clipGradients(self, model, device):
        if self.mask == None:
            return model
        if self.chi == 1:
            return model
        index = 0
        self.mask = self.mask.to(device)
        for m in model.modules():
            m = m.to(device)
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                # If mask entry is 0, multiply gradients by chi
                m.weight.grad.data.mul_(self.mask[index:(index+size)] + self.chi * (1 - self.mask[index:(index+size)]))
                m.bias.grad.data.mul_(self.mask[index:(index+size)] + self.chi * (1 - self.mask[index:(index+size)]))

                index += size
        
        return model


class EarlyBERTGradient(EarlyBirdGradient):

    def __init__(self, chi=1):
        super().__init__(chi)
    
    def clipGradients(self, model, device):
        if self.mask == None:
            return model
        if self.chi == 1:
            return model
        index = 0
        self.mask = self.mask.to(device)
        for m in model.modules():
            m = m.to(device)
            if isinstance(m, bert.modeling_bert.BertSelfAttention):
                size = m.query.weight.data.flatten().shape[0]
                # If mask entry is 0, multiply gradients by chi
                m.query.weight.grad.data.mul_(self.mask[index:(index+size)] + self.chi * (1 - self.mask[index:(index+size)]))
                m.query.bias.grad.data.mul_(self.mask[index:(index+size)] + self.chi * (1 - self.mask[index:(index+size)]))

                index += size

                size = m.key.weight.data.flatten().shape[0]
                # If mask entry is 0, multiply gradients by chi
                m.key.weight.grad.data.mul_(self.mask[index:(index+size)] + self.chi * (1 - self.mask[index:(index+size)]))
                m.key.bias.grad.data.mul_(self.mask[index:(index+size)] + self.chi * (1 - self.mask[index:(index+size)]))
        
        return model


                