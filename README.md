# WASSA-2023

# Useful file description
- **preprocessing.ipynb**: Convert initial pre-processing, such as removing missing values and converting numerical demographic and personal information to textual data. In addition, summarising and paraphrasing texts.
- **combine_data.ipynb**: Combine train set and dev set to be used as train set before testing on the test set
- **main_methods.py**: Methods written for training and testing for any tasks, such as empathy, distress, etc.
- **main_<task>**: Notebook from where I call the methods to make predictions as well as hyperparameter tuning.

# Prepare submission file
`zip predictions.zip predictions_EMP.tsv predictions_IRI.tsv predictions_PER.tsv -v`