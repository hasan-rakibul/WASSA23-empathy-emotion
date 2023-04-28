# Which track?
- CONV: predicting the perceived empathy, emotion polarity and emotion intensity at the speech-turn-level in a conversation
    - can use article text as well by cross-referencing article_id
    - converstion texts are sequence w.r.t. turn_id

# To do:
- data augmentation through Google API / paraphrase tool
- sentiment analysis checkpoint

# Results
- essay from empathy better than incorporating article texts
    - essay-article better than article-essay


- cardiffnlp/twitter-roberta-base-sentiment-latest has longer token size but couldn't use


Use prompt: generate set of text desccription of demographic information


hyperparameter tuning:
learning rate
batch size
dropout if available 


combine train and dev set and do cross-validation -- done
-- team name: Curtin OCAI
-- team participants:
Md Rakibul Hasan
Md Zakir Hossain
Shafin Rahman
Tom Gedeon



CV to find best hyperparameter --> train-dev merge --> train the model --> predict on test set
