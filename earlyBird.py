import torch
import torch.nn as nn
import copy
from transformers.models import bert, gemma

class EarlyBird():
    def __init__(self, ratio, epoch_keep=5, threshold=0.1):
        self.ratio = ratio
        self.threshold = threshold
        self.epoch_keep = epoch_keep
        self.masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]
        self.prevMaskDistance = 1e9

    def pruning(self, model):
        total = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * self.ratio)
        thre = y[thre_index]
        # print('Pruning threshold: {}'.format(thre))

        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
                index += size

        # print('Pre-processing Successful!')
        return mask

    def put(self, mask):
        if len(self.masks) < self.epoch_keep:
            self.masks.append(mask)
        else:
            self.masks.pop(0)
            self.masks.append(mask)

    def cal_dist(self):
        if len(self.masks) == self.epoch_keep:
            for i in range(len(self.masks)-1):
                mask_i = self.masks[-1]
                mask_j = self.masks[i]
                self.dists[i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0)
            return True
        else:
            return False
    
    def get_mask_distance(self):
        return sum(self.dists) / len(self.dists)

    def early_bird_emerge(self, model):
        mask = self.pruning(model)
        self.put(mask)
        flag = self.cal_dist()
        if flag == True:
            print("Mask Distance: " + str(max(self.dists)))
            for i in range(len(self.dists)):
                if self.dists[i] > self.threshold:
                    return False
            return True
        else:
            return False
    
    def compute_mask_distance(self, mask):
        total_dists = [1 for i in range(1, self.epoch_keep)]
        for i in range(len(self.masks)-1):
            mask_i = mask
            mask_j = self.masks[i]
            total_dists[i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0)
        
        return max(total_dists)
        
    def reset_earlyBird(self):
        self.masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]

class EarlyGemma(EarlyBird):
    def __init__(self, ratio, epoch_keep=5, threshold=0.1):
        super().__init__(ratio, epoch_keep, threshold)
    
    def pruning(self, model):
        total = 0
        for m in model.modules():
            if isinstance(m, gemma.modeling_gemma.GemmaSdpaAttention):
                total += m.q_proj.weight.data.flatten().shape[0]
                total += m.k_proj.weight.data.flatten().shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, gemma.modeling_gemma.GemmaSdpaAttention):
                size = m.q_proj.weight.data.flatten().shape[0]
                bn[index:(index+size)] = m.q_proj.weight.data.flatten().abs().clone()
                index += size
                size = m.k_proj.weight.data.flatten().shape[0]
                bn[index:(index+size)] = m.k_proj.weight.data.flatten().abs().clone()
                index += size


        y, i = torch.sort(bn)
        thre_index = int(total * self.ratio)
        thre = y[thre_index]
        # print('Pruning threshold: {}'.format(thre))

        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, gemma.modeling_gemma.GemmaSdpaAttention):
                size = m.q_proj.weight.data.flatten().numel()
                weight_copy = m.q_proj.weight.data.flatten().abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                size = m.k_proj.weight.data.flatten().numel()
                weight_copy = m.k_proj.weight.data.flatten().abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
                index += size

        # print('Pre-processing Successful!')
        return mask

class EarlyBERT(EarlyBird):
    def __init__(self, ratio, epoch_keep=5, threshold=0.1):
        super().__init__(ratio, epoch_keep, threshold)
    
    def pruning(self, model):
        total = 0
        for m in model.modules():
            if isinstance(m, bert.modeling_bert.BertSelfAttention):
                total += m.query.weight.data.flatten().shape[0]
                total += m.key.weight.data.flatten().shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, bert.modeling_bert.BertSelfAttention):
                size = m.query.weight.data.flatten().shape[0]
                bn[index:(index+size)] = m.query.weight.data.flatten().abs().clone()
                index += size
                size = m.key.weight.data.flatten().shape[0]
                bn[index:(index+size)] = m.key.weight.data.flatten().abs().clone()
                index += size


        y, i = torch.sort(bn)
        thre_index = int(total * self.ratio)
        thre = y[thre_index]
        # print('Pruning threshold: {}'.format(thre))

        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, bert.modeling_bert.BertSelfAttention):
                size = m.query.weight.data.flatten().numel()
                weight_copy = m.query.weight.data.flatten().abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                size = m.key.weight.data.flatten().numel()
                weight_copy = m.key.weight.data.flatten().abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
                index += size

        # print('Pre-processing Successful!')
        return mask

