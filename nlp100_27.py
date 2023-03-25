import re

pattern = r'^\{\{基礎情報(.*?)^\}\}'
with open("uk.txt", mode="r")as f:
    data = f.read()
    template = re.findall(pattern, data, re.MULTILINE + re.DOTALL) #基礎情報を抽出
    pattern = r'\|(.+?)\s*=\s*(.+?)(?=\n\|)|(?=\n\})' #"x = y"のxとy
    field_value = re.findall(pattern, template[0], re.MULTILINE + re.DOTALL) #フィールド名と値を抽出
    keys = [re.sub('[*[\]]', '', field_value[i][0]) for i in range(len(field_value))] #keyのマークアップの除去
    values = [re.sub('[*[\]]', '', field_value[i][1]) for i in range(len(field_value))] #valueのマークアップの除去
    basicinfo_removed = {key:value for key,value in zip(keys, values)} #辞書型に
    print(basicinfo_removed)

#13.4身近なライブラリに親しむ,p.172