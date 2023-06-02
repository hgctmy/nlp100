from sklearn.metrics import accuracy_score

semantic = []
semantic_result = []
syntactic = []
syntactic_result = []

with open("questions-words.similar.txt", mode="r")as f:
    for line in f:
        if line.startswith(": gram"):
            # 意味的アナロジーはカテゴリーにgramがついていないもの，文法的アナロジーはgramがついているものらしい．
            # したがってgramflagによって意味的アナロジーか文法的アナロジーかを区別する．
            gramflag = 1
        elif line.startswith(":"):
            gramflag = 0
        elif gramflag == 0:  # 意味的
            words = line.split()
            semantic.append(words[3])  # 答え
            semantic_result.append(words[4])  # 64の結果
        else:  # 文法的
            words = line.split()
            syntactic.append(words[3])
            syntactic_result.append(words[4])

print(accuracy_score(semantic, semantic_result))  # 意味的アナロジーの64の結果の正解率
print(accuracy_score(syntactic, syntactic_result))  # 文法的アナロジーの64の結果の正解率

'''
% python nlp100_65.py
0.17352576389671892
0.2253864168618267
'''
