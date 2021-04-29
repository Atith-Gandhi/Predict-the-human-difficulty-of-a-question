import nltk
nltk.download('punkt')

import json
import requests
import random

college = ['College', 'easy_college', 'regular_college', 'hard_college']
school = ['HS', 'hard_high_school', 'easy_high_school', 'regular_high_school', 'national_high_school']

train_data = requests.get('https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json').json()
random.shuffle(train_data['questions'])

train_processed = {'questions': []}

count_HS = 0
count_college = 0

for i in range(0, len(train_data['questions'])):
    if train_data['questions'][i]['difficulty'] in college:
        train_processed['questions'].append({'text': nltk.tokenize.sent_tokenize(train_data['questions'][i]['text'])[-1], 'difficulty': 'College'})
        count_college = count_college + 1
    elif train_data['questions'][i]['difficulty'] in school:
        train_processed['questions'].append({'text': nltk.tokenize.sent_tokenize(train_data['questions'][i]['text'])[-1], 'difficulty': 'School'})
        count_HS = count_HS + 1
        
test_data = requests.get('https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.test.2018.04.18.json').json()
random.shuffle(test_data['questions'])

test_processed = {'questions': []}

count_college = 0
count_HS = 0

for i in range(0, len(test_data['questions'])):
    if test_data['questions'][i]['difficulty'] in college:
        test_processed['questions'].append({'text': nltk.tokenize.sent_tokenize(test_data['questions'][i]['text'])[-1], 'difficulty': 'College'})
        count_college = count_college + 1
    elif test_data['questions'][i]['difficulty'] in school :
        test_processed['questions'].append({'text': nltk.tokenize.sent_tokenize(test_data['questions'][i]['text'])[-1], 'difficulty': 'School'})
        count_HS = count_HS + 1

with open('quanta_train.json', 'w') as outfile:
    json.dump(train_processed, outfile)
with open('quanta_test.json', 'w') as outfile:
    json.dump(test_processed, outfile)