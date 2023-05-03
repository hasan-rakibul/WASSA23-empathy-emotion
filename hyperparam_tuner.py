import os
import torch
import transformers as trf
import numpy as np
import optuna
import plotly
from functools import partial

from evaluation import pearsonr, calculate_pearson
from main_methods import tokenise, load_tokenised_data

NUM_EPOCH = 35

# train_filename = "train_dev_paraphrased.csv"

# validation by dev set only
train_filename = "train_train_paraphrased.csv"
dev_filename = "dev_summarised.csv"

# Chosen features
feature_1 = 'demographic_essay'
feature_2 = 'article'

checkpoint = "bert-base-uncased"

tokeniser = trf.AutoTokenizer.from_pretrained(checkpoint)

# data collator due to variable max token length per batch size
data_collator = trf.DataCollatorWithPadding(tokenizer = tokeniser)


def objective(trial, task):    
    # Tuning hyperparams:
    LEARNING_RATE = trial.suggest_float("LEARNING_RATE", 1e-05, 1e-04, log=True)
    BATCH_SIZE = trial.suggest_int("BATCH_SIZE", 2, 8)
    SEED = trial.suggest_int("SEED", 1, 100)
    # checkpoint = trial.suggest_categorical("checkpoint", ("bert-base-uncased", "albert-base-v2"))
    
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) # just being extra cautious
    np.random.seed(SEED)
    
    model = trf.AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    ## Split train-test from thw whole train-dev dataset
    # train_dev = load_tokenised_data(filename=os.path.join("./processed_data", train_filename), tokenise_fn=tokenise, train_test="train")
    # train_portion = int(len(train_dev) * 0.8)
    # validation_portion = len(train_dev) - train_portion
    # train_subset, val_subset = torch.utils.data.random_split(train_dev, [train_portion, validation_portion])
    
    # Training by train set only
    train_subset = load_tokenised_data(filename=os.path.join("./processed_data", train_filename), task=task, tokenise_fn=tokenise, train_test="train")
    
    train_dataloader = torch.utils.data.DataLoader(
        train_subset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
    ) 
    
    training_steps = NUM_EPOCH * len(train_dataloader)
    lr_scheduler = trf.get_scheduler(
        "linear",
        optimizer=opt,
        num_warmup_steps=0,
        num_training_steps=training_steps
    )
    
    # Evaluation data loader
    val_subset = load_tokenised_data(filename=os.path.join("./processed_data", dev_filename), task=task, tokenise_fn=tokenise, train_test="train") #treain_test="train" ensures output labesl are also passed
    validation_dataloader = torch.utils.data.DataLoader(
        val_subset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=data_collator
    )
    
    model.train()
    for epoch in range(NUM_EPOCH):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()    
            opt.step()
            lr_scheduler.step()
            opt.zero_grad()

        # Evaluation   
        model.eval()
        y_true =[]
        y_pred = []

        for batch in validation_dataloader:
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

            y_true.extend(batch['labels'].tolist())
            batch_pred = [item for sublist in outputs.logits.tolist() for item in sublist]  #convert 2D list to 1D
            y_pred.extend(batch_pred)
        
        pearson_r = pearsonr(y_true, y_pred)
            
        trial.report(pearson_r, epoch)
            
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return pearson_r

def optuna_tuner(task):
    """
    Run optuna study trial and generate plots
    """
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=28),
        pruner=optuna.pruners.MedianPruner(),
        direction="maximize"
    )
    
    objective_param = partial(objective, task=task) #sending parameters to the objective function
    study.optimize(objective_param, n_trials=100, show_progress_bar=True)

    trial_results = study.trials_dataframe() #trial results as a dataframe
    trial_results.to_csv("trial_results_" + task + ".csv")

    print(f"Best Pearson r: {study.best_value}")
    print(f"Best parameter: {study.best_params}")
    
    fig_1 = optuna.visualization.plot_slice(study)
    fig_1.show()
    fig_1.write_image("./prediction/" + task + "-param-plots.pdf")

    fig_2 = optuna.visualization.plot_param_importances(study)
    fig_2.show()
    fig_2.write_image("./prediction/" + task + "-param-importance.pdf")