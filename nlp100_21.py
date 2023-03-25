import re

pattern = r'^(.*\[\[Category:.*\]\].*)$' #"[[Category:ほにゃらら]]ほにゃらら"というパターン
with open("ukcategory.txt", mode="w")as fw, open("uk.txt", mode="r")as fr:
    data = fr.read()
    fw.write('\n'.join(re.findall(pattern, data, re.MULTILINE))) #Categoryを見つけてuk.txtに出力

#6.6コードの意図を書く,p.76