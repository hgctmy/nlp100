import requests
import re

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

pattern = r'^\{\{基礎情報(.*?)^\}\}'
with open("uk.txt", mode="r")as f:
    data = f.read()
    template = re.findall(pattern, data, re.MULTILINE + re.DOTALL)  # 基礎情報を抽出
    pattern = r'\|(.+?)\s*=\s*(.+?)(?=\n\|)|(?=\n\})'
    field_value = dict(re.findall(pattern, template[0], re.MULTILINE + re.DOTALL))  # フィールド名と値を抽出
    picture_file = field_value["国旗画像"].replace(' ', '_')  # 国旗画像のファイル名
    PARAMS = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": f"File:{picture_file}",
        "iiprop": "url"
    }
    R = S.get(url=URL, params=PARAMS)  # Wikipediaの国旗画像の情報を取得
    result = re.search(r'"url":"(.*?)"', R.text).group(1)  # urlを抽出
    print(result)

    # 4.4縦の線をまっすぐにする,p.47
