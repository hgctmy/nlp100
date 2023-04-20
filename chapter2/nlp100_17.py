import pandas as pd

df = pd.read_table("popular-names.txt", header=None)
print(set(df.iloc[:, 0]))

# 9.1変数を削除する,p.112

'''
% python nlp100_17.py
{'Alexis', 'Chloe', 'Helen', 'Nancy', 'Minnie', 'Mildred', 'Florence', 'Virginia', 'Carolyn', 'Lisa', 'Rachel', 'Heather', 'Mia', 'Bessie', 'Thomas', 'Margaret', 'Liam', 'Linda', 'Deborah', 'Logan', 'Laura', 'Doris', 'Betty', 'Andrew', 'Richard', 'Alexander', 'Jessica', 'Ethan', 'Lauren', 'Jason', 'Brittany', 'Patricia', 'Charlotte', 'Julie', 'Stephanie', 'Aiden', 'Amanda', 'Shirley', 'Ava', 'Christopher', 'Karen', 'Abigail', 'Brian', 'Kimberly', 'Tammy', 'Jayden', 'Olivia', 'Sandra', 'Robert', 'Sarah', 'Marie', 'Angela', 'Nicole', 'Alice', 'Justin', 'Noah', 'Rebecca', 'Austin', 'Amelia', 'Oliver', 'Judith', 'Steven', 'Anna', 'Lori', 'Dorothy', 'Emma', 'Michael', 'George', 'Evelyn', 'Harper', 'Mason', 'Lucas', 'Sharon', 'Ida', 'Cynthia', 'Gary', 'Elijah', 'Brandon', 'Emily', 'Henry', 'John', 'Donald', 'Bertha', 'James', 'Lillian', 'Edward', 'Frank', 'Joseph', 'Hannah', 'Michelle', 'Jennifer', 'Daniel', 'Charles', 'Sophia', 'Isabella', 'Pamela', 'Mary', 'Nicholas', 'Scott', 'Kathleen', 'Elizabeth', 'Susan', 'Samantha', 'Walter', 'Tracy', 'Benjamin', 'Crystal', 'Kelly', 'Jeffrey', 'Debra', 'Madison', 'Annie', 'Barbara', 'Donna', 'Megan', 'Matthew', 'Taylor', 'David', 'Carol', 'Joan', 'Melissa', 'Ashley', 'Ronald', 'Tyler', 'Ethel', 'Anthony', 'Larry', 'Clara', 'William', 'Jacob', 'Mark', 'Amy', 'Frances', 'Joshua', 'Harry', 'Ruth'}

cut -f 1 popular-names.txt | sort | uniq
各行の一項目目を切り取る sortコマンドでソート、uniqコマンドで重複削除
'''
