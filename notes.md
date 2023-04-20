# Deadlines
- Evaluation: April 15, 2023

# Which track?
- CONV: predicting the perceived empathy, emotion polarity and emotion intensity at the speech-turn-level in a conversation
    - can use article text as well by cross-referencing article_id
    - converstion texts are sequence w.r.t. turn_id
- EMP: predicting both the empathy concern and the personal distress at the essay-level
    - predictions_EMP.tsv
    - For the EMP subtask, the first column the prediction values for empathy, and the second column the prediction values for distress

To do:
- data augmentation through Google API
- sentiment analysis checkpoint
- incorporate dev set to test and make full training on the training set

# Results
- learning rate of 2e-5 decrease performance than default (5e-5)
- essay from empathy better than incorporating article texts
    - essay-article better than article-essay
- BERT-base-uncased (0.62) whereas DisTilBert (0.64)


- cardiffnlp/twitter-roberta-base-sentiment-latest has longer token size but couldn't use


Use prompt: generate set of text desccription of demographic information


hyperparameter tuning:
learning rate
batch size
dropout if available 


combine train and dev set and do cross-validation -- done
-- team name: Curtin OCAI

CV to find best hyperparameter --> train-dev merge --> train the model --> predict on test set
