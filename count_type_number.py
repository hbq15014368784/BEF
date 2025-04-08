import json

with open('qid2type_cp2_test.json','r') as file:
    data = json.load(file)

counter = sum(1 for value in data.values() if value == "other")

print(f'The count of "other" category is: {counter}')