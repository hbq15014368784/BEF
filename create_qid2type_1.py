import json


with open('data/vqacp_v2_test_annotations.json', 'r') as f:
    questions_json = json.load(f)

result_dict = {}


yn_num=0
number_num=0
other_num=0
for question_data in questions_json:

    question_id = question_data["question_id"]
    question_type = question_data["answer_type"]

    if question_type=="yes/no":
        yn_num+=1
    elif question_type=="number":
        number_num += 1
    else:
        other_num += 1

    result_dict[question_id] = question_type

print(f"the number of y/n question:{yn_num}")
print(f"the number of number question:{number_num}")
print(f"the number of other question:{other_num}")

result_json = json.dumps(result_dict, indent=2)


with open('qid2type_cp2_test_1.json', 'w', encoding='utf-8') as output_file:
    output_file.write(result_json)

print("Question types have been saved to question_types.json.")
