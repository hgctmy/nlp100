import re

pattern = r'^(.*\[\[Category:.*\]\].*)$'
with open("ukcategory.txt", mode="w")as f1:
    with open("uk.txt", mode="r")as f2:
        data = f2.read()
        f1.write('\n'.join(re.findall(pattern, data, re.MULTILINE)))