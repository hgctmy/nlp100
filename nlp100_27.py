import re

pattern = r'^\{\{基礎情報(.*?)^\}\}'
with open("uk.txt", mode="r")as f:
    data = f.read()
    template = re.findall(pattern, data, re.MULTILINE + re.DOTALL)
    pattern = r'\|(.+?)\s*=\s*(.+?)(?=\n\|)|(?=\n\})'
    field_value = re.findall(pattern, template[0], re.MULTILINE + re.DOTALL)
    keys = [re.sub('[*[\]]', '', field_value[i][0]) for i in range(len(field_value))]
    values = [re.sub('[*[\]]', '', field_value[i][1]) for i in range(len(field_value))]
    field_value_sub = {key:value for key,value in zip(keys, values)}
    print(field_value_sub)