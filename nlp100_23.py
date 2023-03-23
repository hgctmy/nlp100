import re

pattern = r'^(=+)(.*?)(=+)'
with open("uk.txt", mode="r")as f:
    data = f.read()
    print('\n'.join(i[1]+":"+str(len(i[0])-1) for i in re.findall(pattern, data, re.MULTILINE)))