import re

pattern = r'^\{\{基礎情報(.*?)^\}\}'
with open("uk.txt", mode="r")as f:
    data = f.read()
    template = re.findall(pattern, data, re.MULTILINE + re.DOTALL)
    pattern = r'\|(.+?)\s*=\s*(.+?)(?=\n\|)|(?=\n\})'
    field_value = dict(re.findall(pattern, template[0], re.MULTILINE + re.DOTALL))
    print(field_value)