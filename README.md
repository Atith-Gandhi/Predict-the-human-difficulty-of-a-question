# Predict-the-human-difficulty-of-a-question

- The aim of this project is to create a binary classifier that can distinguish a high school question from a college level question. 
- Four different types of models (Bert, DistilBert, ConvBert, Electra) were trained and tested on quanta dataset.
(https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json). 
- Questions that were having difficulty in ['College', 'easy_college', 'regular_college', 'hard_college'] were considered of College class difficulty level while the questions that were having difficulty in ['HS', 'hard_high_school', 'easy_high_school', 'regular_high_school', 'national_high_school'] were considered of High School level difficulty.
- There are separate codes for training the classifier either on the basis of either last line or the full question.

# Results

### When only last line is considered for classification

| Classifier | Accuracy|
| --------------------------- | --------------------------- |
| BERT | 67.45% |
| ConvBERT | 68.1% |
| DistilBERT | 61.4% |
| ELECTRA | 66.75 |

### When the full question is considered for classification

| Classifier | Accuracy|
| --------------------------- | --------------------------- |
| BERT | 82.6% |
| ConvBERT | 81.2% |
| DistilBERT | 79.95% |
| ELECTRA | 80.05% |

## Pretrained models

Download pre-trained models from [here](https://drive.google.com/drive/folders/18dGwaxI7kx4Yx7gTMTiCbUv2YLxzNPmZ?usp=sharing)

Move the pre-trained models to the models/  folder.

## Steps to run the code

- Download the requirements
1. pip install -r requirements.txt

- Create dataset (creates training and testing data for both last_line and full_question classifiers) 
2. curl https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json > quanta.json
3. python3 create_dataset.py

- Train the model ( if you have already downloaded pre-trained models you can skip step 4)
4. python3 BERT/train_last_line.py or  python3 BERT/train_full_question.py

- Test the classifier
5. python3 BERT/test_last_line.py or python3 BERT/test_full_question.py
