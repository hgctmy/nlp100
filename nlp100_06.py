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
