# Predict-the-human-difficulty-of-a-questions

- The aim of this project was to create a classifier that can distinguish a high school question from a college level question. 
- Four different types of models (Bert, DistilBert, ConvBert, Electra) were trained and tested on quanta dataset.
(https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json). 
- There are separate codes for training the classifier either on the basis of last line or the full question.

## Steps to run the code

- Download the requirements
1. pip install -r requirements.txt
- Create dataset (there are two options either create dataset which has only last lines of each question or create dataset of full questions)
2. python3 create_dataset.py
- Train the model ( if you have downloaded already trained models skip step 3)
3. python3 BERT/train_last_line.py / python3 BERT/train_full_question.py
