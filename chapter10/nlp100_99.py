import streamlit as st
import nlp100_92
import torch
import json
import MeCab
import unidic

if "word_id_dict_en" not in st.session_state:
    with open("word_id_dict_en.json")as f:
        st.session_state["word_id_dict_en"] = json.load(f)

if "word_id_dict_ja" not in st.session_state:
    with open("word_id_dict_ja.json")as f:
        st.session_state["word_id_dict_ja"] = json.load(f)


if "model" not in st.session_state:

    SRC_VOCAB_SIZE = len(st.session_state.word_id_dict_ja)
    TGT_VOCAB_SIZE = len(st.session_state.word_id_dict_en)
    EMB_SIZE = 100  # embeddingの次元
    NHEAD = 4  # マルチヘッドアテンションのヘッドの数
    FFN_HID_DIM = 100  # エンコーダーのフィードフォワードの次元
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    model = nlp100_92.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    model.load_state_dict(torch.load("transformer_weights.pth"))
    st.session_state["model"] = model


st.title("日英翻訳")

button_placeholder = st.empty()


def click(input):
    wakati = MeCab.Tagger('-Owakati')
    st.write(nlp100_92.translate(st.session_state.model, wakati.parse(input)))


with button_placeholder.container():
    input = st.text_area('英語に翻訳したい日本語の文章を入力してください．', height=300)
    st.button('翻訳する', key='b1', on_click=lambda: click(input))
