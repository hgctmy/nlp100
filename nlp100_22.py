import re

pattern = r'^.*\[\[Category:(.*?)(?:\|.*)?\]\]$'
with open("uk.txt", mode="r")as f:
    data = f.read()
    print('\n'.join(re.findall(pattern, data, re.MULTILINE)))