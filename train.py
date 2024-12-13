import torch
from transformers import TrainingArguments, Trainer, TrainerCallback
from earlyBird import EarlyBirdGradient

class WormTransformerTrainer(Trainer):

    def __init__(
        self, *args, ebg: EarlyBirdGradient = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ebg = ebg

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        mask = self.ebg.getMask()
        chi = self.ebg.getChi()

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
        

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

def train_one_epoch_with_clip(model, device, train, optimizer, criterion, epoch_num, ebg):
    model.train()
    running_loss = 0.0
    for i, (inputs, outputs) in enumerate(train):
        inputs, outputs = inputs.to(device), outputs.to(device)
        chi = ebg.getChi()
        mask = ebg.getMask()
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, outputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        ebg.updateLoss(running_loss / (i + 1))
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

    return 100 * correct / total

def train_one_epoch_transformer(model, train_dataset, tokenizer, compute_metrics):
    training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs=1, save_strategy="no")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return model

def train_one_epoch_transformer_clip(model, train_dataset, tokenizer, compute_metrics, ebg):
    training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs=1, save_strategy="no")

    trainer = WormTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        ebg=ebg
    )

    trainer.train()

    return model

def test_transformer(model, eval_dataset, tokenizer, compute_metrics):
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate(eval_dataset=eval_dataset)
    print("Accuracy: " + str(results['eval_accuracy'] * 100))
    return results['eval_accuracy'] * 100