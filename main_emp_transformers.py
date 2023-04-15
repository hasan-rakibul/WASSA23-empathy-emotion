import pandas as pd
import numpy as np
import transformers as trf
from datasets import Dataset
import torch
from tqdm.auto import tqdm
from evaluation import pearsonr, calculate_pearson

raw_data = pd.read_csv("./essay_article_text_train_dev.csv", index_col=0)

# raw_data.head(2)

"""# Empathy"""

chosen_data = raw_data[['article', 'essay', 'empathy']]


hugging_dataset = Dataset.from_pandas(chosen_data, preserve_index=False)
hugging_dataset = hugging_dataset.train_test_split(test_size = 0.2)

# hugging_dataset['train']['essay'][:5]

# checkpoint = "bert-base-uncased"
# checkpoint = "bhadresh-savani/bert-base-uncased-emotion"
checkpoint = "distilbert-base-uncased"
# checkpoint = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokeniser = trf.AutoTokenizer.from_pretrained(checkpoint)

#padding="longest" can be deferred to do dynamic padding
def tokenise(sentence):
  return tokeniser(sentence["essay"], sentence["article"], truncation=True) 
  # return tokeniser(sentence["essay"], sentence["article"], padding="max_length", max_length=514, truncation=True)   #for Cardiff-emotion one
  # return tokeniser(sentence["essay"], truncation=True) 
  # return tokeniser(sentence["article"], sentence["essay"], truncation=True)

tokenised_hugging_dataset = hugging_dataset.map(tokenise, batched=True)

# tokenised_hugging_dataset

# checking length after tokenisation
# length = []
# for i in range(tokenised_hugging_dataset['train'].num_rows):
#   length.append(len(tokenised_hugging_dataset['train']['input_ids'][i]))

# print(f"Lengths: {length}")

tokenised_hugging_dataset = tokenised_hugging_dataset.remove_columns(["article","essay"]) # no longer required as encoding done
tokenised_hugging_dataset = tokenised_hugging_dataset.rename_column("empathy", "labels") # as huggingface requires
tokenised_hugging_dataset = tokenised_hugging_dataset.with_format("torch")

# tokenised_hugging_dataset

"""# Prediction model"""

BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCH = 3

data_collator = trf.DataCollatorWithPadding(tokenizer = tokeniser)

train_dataloader = torch.utils.data.DataLoader(
    tokenised_hugging_dataset["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)

test_dataloader = torch.utils.data.DataLoader(
    tokenised_hugging_dataset["test"], batch_size=BATCH_SIZE, collate_fn=data_collator
)


prediction_model = trf.AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)
# prediction_model = trf.AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1, ignore_mismatched_sizes=True)

opt = torch.optim.AdamW(prediction_model.parameters(), lr=LEARNING_RATE)

training_steps = NUM_EPOCH * len(train_dataloader)
lr_scheduler = trf.get_scheduler(
    "linear",
    optimizer=opt,
    num_warmup_steps=0,
    num_training_steps=training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
prediction_model.to(device)
print(device)

# criterion = torch.nn.MSELoss()

progress_bar = tqdm(range(training_steps))

prediction_model.train()
for epoch in range(NUM_EPOCH):
  epoch_loss = 0
  num_batches = 0
  for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = prediction_model(**batch)
    loss = outputs.loss
    # loss = criterion(outputs.logits, batch['labels'])
    loss.backward()

    opt.step()
    lr_scheduler.step()
    opt.zero_grad()
    progress_bar.update(1)

    epoch_loss += loss.item()
    num_batches += 1

  avg_epoch_loss = epoch_loss / num_batches
  print(f"Epoch {epoch}: average loss = {avg_epoch_loss}")


"""## Evaluation"""

prediction_model.eval()

predictions = []

for batch in test_dataloader:
  batch = {k: v.to(device) for k, v in batch.items()}
  with torch.no_grad():
    outputs = prediction_model(**batch)
    
  batch_pred = [item for sublist in outputs.logits.tolist() for item in sublist]  #convert 2D list to 1D
  predictions.append(batch_pred)

y_pred = [item for sublist in predictions for item in sublist]  #convert batch-wise 2D list to 1D

# prediction_model.save_pretrained("model")

y_true = hugging_dataset["test"]["empathy"]

# y_pred

# y_true

print(f"Pearson r for empahty: {pearsonr(y_true,y_pred)}")