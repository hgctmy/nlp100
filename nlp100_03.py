import re

str = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
wordlist = re.sub(r'[,.]', '', str)  # ,.を除去
wordlist = str.split()  # スペースで分解
print([len(word) for word in wordlist])  # 単語の文字数リスト

# 9.1変数を削除する,p.112
