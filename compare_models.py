import torch
from datasets import load_dataset, load_metric
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer

# Load the dataset and metrics
dataset = load_dataset('imdb', split='train[:50%]')
metric = load_metric('accuracy')


# To compare the performance of different language models, we can train and evaluate them on the same 
#  task and dataset. For this experiment, let's use the IMDB movie reviews dataset and fine-tune the models 
#  for binary sentiment classification (positive/negative reviews).

# Define the models and tokenizers
models = [
    ('distilbert-base-uncased', DistilBertForSequenceClassification, DistilBertTokenizer),
    ('bert-base-uncased', BertForSequenceClassification, BertTokenizer),
    ('bert-large-uncased', BertForSequenceClassification, BertTokenizer),
    ('bert-base-cased', BertForSequenceClassification, BertTokenizer),
    ('bert-large-cased', BertForSequenceClassification, BertTokenizer),
    ('roberta-base', RobertaForSequenceClassification, RobertaTokenizer),
    ('roberta-large', RobertaForSequenceClassification, RobertaTokenizer),
]

# Fine-tune and evaluate each model
results = []
for model_name, model_class, tokenizer_class in models:
    try:
        # Initialize the tokenizer and model
        tokenizer = tokenizer_class.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)

        # Tokenize the input texts
        def tokenize(batch):
            return tokenizer(batch['text'], padding=True, truncation=True)
        dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))

        # Prepare the dataset for training
        dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        train_dataset = dataset.select(range(0, len(dataset)//2))
        dev_dataset = dataset.select(range(len(dataset)//2, len(dataset)))

        # Train the model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        for epoch in range(3):
            for batch in torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate the model
        model.eval()
        predictions = []
        for batch in torch.utils.data.DataLoader(dev_dataset, batch_size=16):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        accuracy = metric.compute(predictions=predictions, references=dev_dataset['label'])
        print(f'{model_name}: {accuracy:.2f}')
        results.append((model_name, accuracy))
    except Exception as e:
       
