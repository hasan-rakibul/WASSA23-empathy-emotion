import pandas as pd
import numpy as np
import transformers as trf
from datasets import Dataset
import torch
import os
from evaluation import pearsonr

os.environ["TOKENIZERS_PARALLELISM"] = "false" # due to huggingface warning

NUM_EPOCH = 35

#final test
train_filename = "train_dev_paraphrased.csv"
test_filename = "preprocessed_test.csv"

# developement time
# train_filename = "train_train_paraphrased.csv"
# test_filename = "preprocessed_dev.csv"
# # test_filename = "preprocessed_complete_dev.csv"

#Chosen features
feature_1 = 'demographic_essay'
feature_2 = 'article'

checkpoint = "bert-base-uncased"

tokeniser = trf.AutoTokenizer.from_pretrained(checkpoint)

# data collator due to variable max token length per batch size
data_collator = trf.DataCollatorWithPadding(tokenizer = tokeniser)

#padding="longest" can be deferred to do dynamic padding
def tokenise(sentence):
    return tokeniser(sentence[feature_1], sentence[feature_2], truncation=True)

def load_tokenised_data(filename, task, tokenise_fn, train_test):
   
    input_data = pd.read_csv(filename, header=0, index_col=0)
    
    if train_test == "train":
        chosen_data = input_data[[feature_1, feature_2, task]]
    elif train_test == "test":
        chosen_data = input_data[[feature_1, feature_2]]  #test data shouldn't have output label

    hugging_dataset = Dataset.from_pandas(chosen_data, preserve_index=False)

    tokenised_hugging_dataset = hugging_dataset.map(tokenise_fn, batched=True, remove_columns = [feature_1, feature_2])
    
    if train_test == "train":
        tokenised_hugging_dataset = tokenised_hugging_dataset.rename_column(task, "labels") # as huggingface requires
    
    tokenised_hugging_dataset = tokenised_hugging_dataset.with_format("torch")

    return tokenised_hugging_dataset
    

def train_test_wo_acc(task, lr, batch_size, seed):
    """
    train-test pipeline without huggingface accelerator
    """
    print(f"{task} prediction")  #task: "empathy" or "distress" or ...
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # just being extra cautious
    np.random.seed(seed)
    
    model = trf.AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
  
    trainset = load_tokenised_data(filename=os.path.join("./processed_data", train_filename), task=task, tokenise_fn=tokenise, train_test="train")
       
    trainloader = torch.utils.data.DataLoader(
        trainset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    
    training_steps = NUM_EPOCH * len(trainloader)
    lr_scheduler = trf.get_scheduler(
        "linear",
        optimizer=opt,
        num_warmup_steps=0,
        num_training_steps=training_steps
    )
    
    # evaluation data loader
    testset = load_tokenised_data(filename=os.path.join("./processed_data", test_filename), task=task, tokenise_fn=tokenise, train_test="test")
    testloader = torch.utils.data.DataLoader(
        testset, shuffle=False, batch_size=batch_size, collate_fn=data_collator
    )
    
    model.train()
    for epoch in range(0, NUM_EPOCH):        
        epoch_loss = 0
        num_batches = 0

        # Iterate over the DataLoader for training data
        for batch in trainloader:
            # Perform forward pass
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            opt.step()
            lr_scheduler.step()         
            opt.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1

        # Process is complete.
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}: average loss = {avg_epoch_loss}")    
            
        # Starting evaluation
        model.eval()
        y_pred = []

        for batch in testloader:
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

            batch_pred = [item for sublist in outputs.logits.tolist() for item in sublist]  #convert 2D list to 1D
            y_pred.extend(batch_pred)
  
    y_pred_df = pd.DataFrame({task: y_pred})
    filename = "./tmp/predictions_" + task + ".tsv"
    y_pred_df.to_csv(filename, sep='\t', header=False, index=False)
    
# def train_test_wo_acc_print_val_score(task, lr, batch_size, seed):
#     """
#     train-test pipeline without huggingface accelerator
#     print validation performance in each step
#     """
#     print(f"{task} prediction")  #task: "empathy" or "distress" or ...
    
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # just being extra cautious
#     np.random.seed(seed)
    
#     model = trf.AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)
    
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model.to(device)
    
#     opt = torch.optim.AdamW(model.parameters(), lr=lr)
  
#     trainset = load_tokenised_data(filename=os.path.join("./processed_data", train_filename), task=task, tokenise_fn=tokenise, train_test="train")
       
#     trainloader = torch.utils.data.DataLoader(
#         trainset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
#     )
    
#     training_steps = NUM_EPOCH * len(trainloader)
#     lr_scheduler = trf.get_scheduler(
#         "linear",
#         optimizer=opt,
#         num_warmup_steps=0,
#         num_training_steps=training_steps
#     )
    
#     # evaluation data loader - "train" mode ensure output label
#     testset = load_tokenised_data(filename=os.path.join("./processed_data", test_filename), task=task, tokenise_fn=tokenise, train_test="train")
#     testloader = torch.utils.data.DataLoader(
#         testset, shuffle=False, batch_size=batch_size, collate_fn=data_collator
#     )
    
#     model.train()
#     for epoch in range(0, NUM_EPOCH):        
#         epoch_loss = 0
#         num_batches = 0

#         # Iterate over the DataLoader for training data
#         for batch in trainloader:
#             # Perform forward pass
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(**batch)
#             loss = outputs.loss

#             loss.backward()
#             opt.step()
#             lr_scheduler.step()         
#             opt.zero_grad()
            
#             epoch_loss += loss.item()
#             num_batches += 1

#         # Process is complete.
#         avg_epoch_loss = epoch_loss / num_batches
#         print(f"Epoch {epoch+1}: average loss = {avg_epoch_loss}")    
            
#         # Starting evaluation with each training epoch
#         model.eval()
#         y_pred = []
#         y_true=[]

#         for batch in testloader:
#             with torch.no_grad():
#                 batch = {k: v.to(device) for k, v in batch.items()}
#                 outputs = model(**batch)

#             batch_pred = [item for sublist in outputs.logits.tolist() for item in sublist]  #convert 2D list to 1D
#             y_pred.extend(batch_pred)
#             y_true.extend(batch['labels'].tolist())
            
#         pearson_r = pearsonr(y_true, y_pred)
#         print(pearson_r)
  
#     y_pred_df = pd.DataFrame({task: y_pred})
#     filename = "./prediction/predictions_" + task + ".tsv"
#     y_pred_df.to_csv(filename, sep='\t', header=False, index=False)
    
    
#### Using Huggingface accelerator
# from accelerate import Accelerator
# from accelerate import notebook_launcher
# def train_test(model, task, lr, batch_size):
#     accelerator = Accelerator()
    
#     accelerator.print(f"{task} prediction")  #task: "empathy" or "distress" or ...
    
#     opt = torch.optim.AdamW(model.parameters(), lr=lr)
  
#     trainset = load_tokenised_data(filename=os.path.join("./processed_data", train_filename), task=task, tokenise_fn=tokenise, train_test="train")
       
#     trainloader = torch.utils.data.DataLoader(
#         trainset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
#     )
    
#     training_steps = NUM_EPOCH * len(trainloader)
#     lr_scheduler = trf.get_scheduler(
#         "linear",
#         optimizer=opt,
#         num_warmup_steps=0,
#         num_training_steps=training_steps
#     )

#     trainloader, model, opt = accelerator.prepare(
#         trainloader, model, opt    
#     )
    
#     # evaluation data loader
#     testset = load_tokenised_data(filename=os.path.join("./processed_data", test_filename), task=task, tokenise_fn=tokenise, train_test="test")
#     testloader = torch.utils.data.DataLoader(
#         testset, shuffle=False, batch_size=batch_size, collate_fn=data_collator
#     )
    
#     model.train()
#     for epoch in range(0, NUM_EPOCH):        
#         epoch_loss = 0
#         num_batches = 0

#         # Iterate over the DataLoader for training data
#         for batch in trainloader:
#             # Perform forward pass
#             outputs = model(**batch)
            
#             loss = outputs.loss

#             accelerator.backward(loss)
        
#             opt.step()
#             lr_scheduler.step()
            
#             opt.zero_grad()
            
#             epoch_loss += loss.item()
#             num_batches += 1

#         # Process is complete.
#         avg_epoch_loss = epoch_loss / num_batches
#         accelerator.print(f"Epoch {epoch+1}: average loss = {avg_epoch_loss}")    
            
#         # Starting evaluation
#         model.eval()
#         y_pred = []

#         for batch in testloader:
#             with torch.no_grad():
#                 outputs = model(**batch)

#             batch_pred = [item for sublist in outputs.logits.tolist() for item in sublist]  #convert 2D list to 1D
#             y_pred.extend(batch_pred)
  
#     y_pred_df = pd.DataFrame({task: y_pred})
#     filename = "./prediction/predictions_" + task + ".tsv"
#     y_pred_df.to_csv(filename, sep='\t', header=False, index=False)

# def final_prediction(task, lr, batch, seed):

#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
    
#     model = trf.AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)

#     notebook_launcher(train_test, (model,task,lr,batch), num_processes=torch.cuda.device_count())