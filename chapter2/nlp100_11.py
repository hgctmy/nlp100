with open("tab2space.txt", mode="w") as fw, open("popular-names.txt", mode="r") as fr:
    for line in fr:
        fw.write(line.replace("\t", " "))

# 7.7ネストを浅くする,p.93

'''
ファイルの初め10行
Mary F 7065 1880
Anna F 2604 1880
Emma F 2003 1880
Elizabeth F 1939 1880
Minnie F 1746 1880
Margaret F 1578 1880
Ida F 1472 1880
Alice F 1414 1880
Bertha F 1320 1880
Sarah F 1288 1880

sed -e 's/\t/ /g' ./popular-names.txt
'''
