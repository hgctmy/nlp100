import re

pattern = r'^(=+)(.*?)(=+)'  # 連続した"="とそれらによって挟まれた文字列
with open("uk.txt", mode="r")as f:
    data = f.read()
    # [一つ以上の"=",セクション名,一つ以上の"="]を要素としたリスト
    section = re.findall(pattern, data, re.MULTILINE)
    print('\n'.join(i[1]+":"+str(len(i[0])-1)for i in section))  # "セクション名:レベル"を出力

# 7.7ネストを浅くする,p.93
