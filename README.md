# WASSA-2023


# Useful file description
- **combine_data.ipynb**: For essay-level tasks, combine (1) dev set with corresponding labels and (2) train set and dev set
- **preprocessing.ipynb**: For essay-level-tasks, convert initial pre-processing, such as removing missing values and converting numerical demographic and personal information to textual data. In addition, summarising and paraphrasing texts.
- **preprocessing_CONV.ipynb**: Data combine and preprocessing for CONV task
- **main_methods.py**: Methods written for training and testing for any essay-level tasks, such as empathy, distress, etc.
- **hyperparam_tuner.py**: Methods written for optuna hyperparam tuner.
- **main_\<task\>**: Notebook from where I call the methods to make predictions as well as hyperparameter tuning on essay-level tasks. For CONV task, no separate py files are used as method and tuner since it's not required for any other tasks.

# How to run
- Create directories
	- `dataset/`
		- `dataset/dev`
		- `dataset/test`
	- `processed_data/`
	- `predictions/`
	- `tmp/`
- Download the dataset from (the official WASSA 2023 competition site at CodaLab)[https://codalab.lisn.upsaclay.fr/competitions/11167]
	- Keep training set and `article_adobe_AMT.csv` in `dataset/`folder
	- Keep dev set at `dataset/dev/` folder
	- Keep test set at `dataset/test/` folder
- Essay-level tasks
	- `combine_data.ipynb`
	- `preprocessing.ipynb`
	- `main_\<task\>.ipynb`
- Conversation-level tasks
	- `preprocessing_CONV.ipynb`
	- `main_CONV.ipynb`

