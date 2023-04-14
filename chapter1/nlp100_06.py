str1 = "paraparaparadise"
str2 = "paragraph"

X = set([(str1[i], str1[i+1]) for i in range(len(str1)-1)])  # 文字bigram (重複なし）
Y = set([(str2[i], str2[i+1]) for i in range(len(str2)-1)])

print("和:", X | Y)
print("積:", X & Y)
print("差:", X-Y)
print("Xにseが含まれるか:", ('s', 'e') in X)
print("Yにseが含まれるか:", ('s', 'e') in Y)

# 13.4身近なライブラリに親しむ,p.172

'''
% python nlp100_06.py
和: {('d', 'i'), ('a', 'g'), ('i', 's'), ('s', 'e'), ('a', 'r'), ('a', 'p'), ('a', 'd'), ('p', 'h'), ('r', 'a'), ('g', 'r'), ('p', 'a')}
積: {('a', 'r'), ('a', 'p'), ('p', 'a'), ('r', 'a')}
差: {('s', 'e'), ('d', 'i'), ('a', 'd'), ('i', 's')}
Xにseが含まれるか: True
Yにseが含まれるか: False
'''
