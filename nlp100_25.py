import re

pattern = r'^\{\{基礎情報(.*?)^\}\}'  # "基礎情報"以降
with open("uk.txt", mode="r")as f:
    data = f.read()
    template = re.findall(pattern, data, re.MULTILINE + re.DOTALL)
    pattern = r'\|(.+?)\s*=\s*(.+?)(?=\n\|)|(?=\n\})'  # "x = y"のxとy
    basicinfo = dict(re.findall(pattern, template[0], re.MULTILINE + re.DOTALL))  # フィールド名:値
    print(basicinfo)

# 6.1コメントを簡潔にしておく,p.72
