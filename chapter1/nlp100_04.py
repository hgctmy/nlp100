str = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
str.replace(".", "")
wordlist = str.split()
element = {}
for i, word in enumerate(wordlist):  # 番号付きリスト
    if i+1 in [1, 5, 6, 7, 8, 9, 15, 16, 19]:
        element[word[:1]] = i+1  # keyは単語の一文字目、valueは番号
    else:
        element[word[:2]] = i+1  # keyは単語頭二文字
print(element)

# 13.4身近なライブラリに親しむ,p.172

'''
% python nlp100_04.py
{'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mi': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20}
'''
