import re

pattern = r'^\[\[ファイル:(.*?)\|'  # ファイル名
with open("uk.txt", mode="r")as f:
    data = f.read()
    # ファイル参照を探してそのファイル名を出力
    print('\n'.join(re.findall(pattern, data, re.MULTILINE)))

# 6.6コードの意図を書く,p.76
