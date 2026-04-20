# with open('input.txt', 'r', encoding='utf-8') as f:
#     text = f.read().split()
# print("text",text)
# total_data=[]
# for single_data in text:
#     print(single_data)
#     total_data.append(single_data)

# print("Unique data is",total_data)

# unique_data=[]

# for single_data in total_data:
#     print(single_data)
#     if single_data not in unique_data:
#         unique_data.append(single_data)
#     print("Unique data bharring",unique_data)
# print("Unique data is",unique_data)


# idx={}
# final_data=[]
# for i in range(len(unique_data)):
#     print(unique_data[i])
#     new_token= { 'word':unique_data[i], 'id':i}
#     final_data.append(new_token)

# print("Final data is",final_data)


quiz_questions=["What is capital of usa?","What is capital pf nepal?"]
quiz_answer=["DC","Ktm"]

options=[["DC","Texas"],["DC","Texas","Aasd"]]
ans={
    "q":1,
    "ans":1,
    "options":1
}
print(quiz_questions[0])
print(quiz_answer[0])
answer_user=input()

if answer_user=="DC":
    print("You are right!")
else:
    print("wrong!")