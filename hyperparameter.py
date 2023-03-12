import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset, load_metric

# load the dataset
dataset = load_dataset('glue', 'mrpc')

# split the dataset into train, validation and test sets
train_val_dataset, test_dataset = dataset['train'].train_test_split(test_size=0.1)
train_dataset, val_dataset = train_val_dataset.train_test_split(test_size=0.1)

# load the tokenizer and encode the inputs
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
train_encodings = tokenizer(train_dataset['sentence1'], train_dataset['sentence2'], truncation=True, padding=True)
val_encodings = tokenizer(val_dataset['sentence1'], val_dataset['sentence2'], truncation=True, padding=True)
test_encodings = tokenizer(test_dataset['sentence1'], test_dataset['sentence2'], truncation=True, padding=True)

# create PyTorch datasets
class MRPCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MRPCDataset(train_encodings, train_dataset['label'])
val_dataset = MRPCDataset(val_encodings, val_dataset['label'])
test_dataset = MRPCDataset(test_encodings, test_dataset['label'])

# create PyTorch dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# define the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

# define the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=1e-4)
total_steps = len(train_loader) * 9
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# define the loss function and evaluation metric
loss_fn = torch.nn.CrossEntropyLoss()
metric = load_metric('glue', 'mrpc')

# train the model for different hyperparameters
best_val_accuracy = -float('inf')
best_model = None
for lr in [1e-4, 5e-4, 1e-3]:
    for epochs in [5, 7, 9]:
        print(f'Training model with lr={lr} and {epochs} epochs...')
        optimizer.param_groups[0]['lr'] = lr
        for epoch in range(epochs):
            train_loss = 0.0
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            val_accuracy = evaluate(model, val_loader, metric)
            print(f'Epoch {epoch+1}/{epochs}, train_loss={train_loss:.
