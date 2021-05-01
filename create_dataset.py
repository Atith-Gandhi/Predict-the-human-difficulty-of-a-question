import nltk
nltk.download('punkt')
import os
import json
import requests
import random

college = ['College', 'easy_college', 'regular_college', 'hard_college']
school = ['HS', 'hard_high_school', 'easy_high_school', 'regular_high_school', 'national_high_school']

# os.system('curl https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json --output quanta.json')
f = open('quanta.json')
data = json.load(f)
random.shuffle(data['questions'])

last_line_data = []

count_school = 0
count_college = 0

for i in range(0, len(data['questions'])):
    if data['questions'][i]['difficulty'] in college and count_college < 5000:
        last_line_data.append({'text': nltk.tokenize.sent_tokenize(data['questions'][i]['text'])[-1], 'difficulty': 'College'})
        count_college = count_college + 1
    elif data['questions'][i]['difficulty'] in school and count_school < 5000:
        last_line_data.append({'text': nltk.tokenize.sent_tokenize(data['questions'][i]['text'])[-1], 'difficulty': 'School'})
        count_school = count_school + 1


with open('dataset_last_line/quanta_train.json', 'w') as outfile:
    json.dump({'questions': last_line_data[:8000]}, outfile)
with open('dataset_last_line/quanta_test.json', 'w') as outfile:
    json.dump({'questions': last_line_data[8000:]}, outfile)


full_question = []

count_school = 0
count_college = 0

for i in range(0, len(data['questions'])):
    if data['questions'][i]['difficulty'] in college and count_college < 5000:
        full_question.append({'text': data['questions'][i]['text'], 'difficulty': 'College'})
        count_college = count_college + 1
    elif data['questions'][i]['difficulty'] in school and count_school < 5000:
        full_question.append({'text': data['questions'][i]['text'], 'difficulty': 'School'})
        count_school = count_school + 1


with open('dataset_full_question/quanta_train.json', 'w') as outfile:
    json.dump({'questions': full_question[:8000]}, outfile)
with open('dataset_full_question/quanta_test.json', 'w') as outfile:
    json.dump({'questions': full_question[8000:]}, outfile)
