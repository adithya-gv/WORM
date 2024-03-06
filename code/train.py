import torch

def train_one_epoch(model, device, train, optimizer, criterion, epoch_num):
    model.train()
    running_loss = 0.0
    for i, (inputs, outputs) in enumerate(train):
        inputs, outputs = inputs.to(device), outputs.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, outputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 0:
            print(f'[{epoch_num + 1}, {i + 1}] loss: {running_loss / (i + 1)}')
    
    return model
        
def test(model, test, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, outputs in test:
            inputs, outputs = inputs.to(device), outputs.to(device)
            predictions = model(inputs)
            _, predicted = torch.max(predictions, 1)
            total += outputs.size(0)
            correct += (predicted == outputs).sum().item()
    print(f'Accuracy: {100 * correct / total}')