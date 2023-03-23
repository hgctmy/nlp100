import re

pattern = r'^\[\[ファイル:(.*?)\|'
with open("uk.txt", mode="r")as f:
    data = f.read()
    print('\n'.join(re.findall(pattern, data, re.MULTILINE)))