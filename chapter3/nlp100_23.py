import re

pattern = r'^(=+)(.*?)(=+)'  # 連続した"="とそれらによって挟まれた文字列
with open("uk.txt", mode="r")as f:
    data = f.read()
    # [一つ以上の"=",セクション名,一つ以上の"="]を要素としたリスト
    section = re.findall(pattern, data, re.MULTILINE)
    print('\n'.join(i[1]+":"+str(len(i[0])-1)for i in section))  # "セクション名:レベル"を出力

# 7.7ネストを浅くする,p.93

'''
% python nlp100_23.py
国名:1
歴史:1
地理:1
主要都市:2
気候:2
政治:1
元首:2
法:2
内政:2
地方行政区分:2
外交・軍事:2
経済:1
鉱業:2
農業:2
貿易:2
不動産:2
エネルギー政策:2
通貨:2
企業:2
通信:3
交通:1
道路:2
鉄道:2
海運:2
航空:2
科学技術:1
国民:1
言語:2
宗教:2
婚姻:2
移住:2
教育:2
医療:2
文化:1
食文化:2
文学:2
哲学:2
音楽:2
ポピュラー音楽:3
映画:2
コメディ:2
国花:2
世界遺産:2
祝祭日:2
スポーツ:2
サッカー:3
クリケット:3
競馬:3
モータースポーツ:3
野球:3
 カーリング :3
 自転車競技 :3
脚注:1
関連項目:1
外部リンク:1
'''