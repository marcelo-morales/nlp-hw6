import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# set up data loaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# set up model and optimizer
model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# set up loss function
loss_fn = torch.nn.CrossEntropyLoss()

# send model to GPU
device = torch.device('cuda')
model.to(device)

# training loop
train_acc_list = []
val_acc_list = []
for epoch in range(30):
    # set model to training mode
    model.train()
    
    # train on each batch
    train_loss = 0.0
    train_acc = 0.0
    for batch in train_dataloader:
        # move inputs and targets to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device)

        # compute model output
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, targets)

        # perform backpropagation and update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute accuracy
        predictions = torch.argmax(logits, dim=1)
        acc = torch.mean((predictions == targets).float())

        # accumulate loss and accuracy over batches
        train_loss += loss.item()
        train_acc += acc.item()

    # compute average loss and accuracy over all batches
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    # set model to evaluation mode
    model.eval()

    # compute validation accuracy
    val_acc = 0.0
    with torch.no_grad():
        for batch in validation_dataloader:
            # move inputs and targets to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            # compute model output
            logits = model(input_ids, attention_mask)

            # compute accuracy
            predictions = torch.argmax(logits, dim=1)
            acc = torch.mean((predictions == targets).float())

            # accumulate accuracy over batches
            val_acc += acc.item()

    # compute average validation accuracy
    val_acc /= len(validation_dataloader)

    # save accuracy values
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    # print progress
    print(f'Epoch {epoch+1:02d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')

# plot accuracy values
plt.plot(train_acc_list, label='Training accuracy')
plt.plot(val_acc_list, label='Validation accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
