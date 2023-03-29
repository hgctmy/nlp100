with open("tab2space.txt", mode="w") as fw, open("popular-names.txt", mode="r") as fr:
    for line in fr:
        fw.write(line.replace("\t", " "))

# 7.7ネストを浅くする,p.93
