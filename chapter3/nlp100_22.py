import re

# カテゴリ名、つまり"[[Category:x]]"のxの部分
pattern = r'^.*\[\[Category:(.*?)(?:\|.*)?\]\]$'
with open("uk.txt", mode="r")as f:
    data = f.read()
    print('\n'.join(re.findall(pattern, data, re.MULTILINE)))  # カテゴリ名を抽出

# 6.6コードの意図を書く,p.76

'''
% python nlp100_22.py
イギリス
イギリス連邦加盟国
英連邦王国
G8加盟国
欧州連合加盟国
海洋国家
現存する君主国
島国
1801年に成立した国家・領域
'''
