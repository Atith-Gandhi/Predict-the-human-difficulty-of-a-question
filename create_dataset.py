import nltk
nltk.download('punkt')

import json
import requests
import random

college = ['College', 'easy_college', 'regular_college', 'hard_college']
school = ['HS', 'hard_high_school', 'easy_high_school', 'regular_high_school', 'national_high_school']

data = requests.get('https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json').json()
random.shuffle(data['questions'])

data_processed = []

count_school = 0
count_college = 0

for i in range(0, len(data['questions'])):
    if data['questions'][i]['difficulty'] in college and count_college < 5000:
        data_processed.append({'text': nltk.tokenize.sent_tokenize(data['questions'][i]['text'])[-1], 'difficulty': 'College'})
        count_college = count_college + 1
    elif data['questions'][i]['difficulty'] in school and count_school < 5000:
        data_processed.append({'text': nltk.tokenize.sent_tokenize(data['questions'][i]['text'])[-1], 'difficulty': 'School'})
        count_school = count_school + 1


with open('quanta_train.json', 'w') as outfile:
    json.dump({'questions': data_processed[:8000]}, outfile)
with open('quanta_test.json', 'w') as outfile:
    json.dump({'questions': data_processed[8000:]}, outfile)
