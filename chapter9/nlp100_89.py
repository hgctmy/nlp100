import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')


class CustomDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_table(file_path)
        self.feature = data['TITLE']
        self.labels = torch.tensor([0 if x == 'b' else 1 if x == 't' else 2 if x == 'e' else 3 for x in data['CATEGORY']])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = {'text': self.feature[idx], 'labels': self.labels[idx]}
        encodings = tokenizer(feature['text'], padding=True, truncation=True)
        encodings['labels'] = feature['labels']
        return encodings


train_data = CustomDataset("../chapter8/train.txt")
test_data = CustomDataset("../chapter8/test.txt")


# 評価指標の定義
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions[1], axis=1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.to(device)

batch_size = 16
logging_steps = len(train_data) // batch_size
args = TrainingArguments(
    output_dir='./model',
    learning_rate=1e-5,
    num_train_epochs=5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy='epoch',
    logging_steps=logging_steps,
    report_to='none'
)


trainer = Trainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data
)

# optimizerはデフォルトのAdamW
trainer.train()
trainer.save_model('.ans89/model')
trainer.evaluate()
preds_output = trainer.predict(test_data)
y_preds = np.argmax(preds_output.predictions[1], axis=1)
print(accuracy_score(test_data.labels, y_preds))

'''
0.9415292353823088
'''
