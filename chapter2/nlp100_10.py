# 行数をカウントする
count = 0
with open("popular-names.txt", mode="r") as f:
    for line in f:
        count += 1
    print(count)

# 5.1コメントするべきではないこと,p.57

'''
% python nlp100_10.py
2780

wc -l ./popular-names.txt
    2779 ./popular-names.txt
'''
