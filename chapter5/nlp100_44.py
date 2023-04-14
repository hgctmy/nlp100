import nlp100_41
import graphviz

sentences = nlp100_41.load_result()
chunks = sentences[2]
g = graphviz.Digraph(format='svg', filename='dependencytree')
for i, chunk in enumerate(chunks):  # 同じ文字列の異なるノードを区別するために番号つける
    if chunk.dst != -1:
        source = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs]) + f'({i})'  # 係元
        destination = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunks[chunk.dst].morphs]) + f'({chunk.dst})'  # 係先
        g.node(source)  # 係元ノードを作成
        g.node(destination)  # 係先ノードを作成
        g.edge(source, destination)  # 係元ノードと係先ノードを繋げる
g.view()

# 13.4身近なライブラリに親しむ,p.172
