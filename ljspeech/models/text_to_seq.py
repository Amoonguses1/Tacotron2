# import IPython
# from IPython.display import Audio
# import tensorboard as tb
# import os
# # 数値演算
# import numpy as np
# import torch
# from torch import nn
# # 音声波形の読み込み
# from scipy.io import wavfile
# # フルコンテキストラベル、質問ファイルの読み込み
# from nnmnkwii.io import hts
# # 音声分析
# import pyworld
# # 音声分析、可視化
# import librosa
# import librosa.display
# # Pythonで学ぶ音声合成
# import ttslearn
# from ttslearn.util import init_seed
# from ttslearn.notebook import get_cmap, init_plot_style, savefig


# # シードの固定
# init_seed(773)
# # visualization config
# cmap = get_cmap()
# init_plot_style()


# dictionary
charcters = "abcdefghijklmnopqrstuvwxyz!'(),-.:;?£ \""
extra_symbols = [
    "^",  # start of sentences
    "$"  # end of sentenses
]
_pad = "~"
symbols = [_pad] + extra_symbols + list(charcters) + [str(i) for i in range(10)]
_symbols_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symobls = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text):
    text = text.lower()
    seq = [ _symbols_to_id["^"] ]
    seq += [ _symbols_to_id[s] if s in _symbols_to_id else 40 for s in text ]
    seq.append(_symbols_to_id["$"])
    return seq


def sequence_to_text(seq):
    return [_id_to_symobls[id_] for id_ in seq]


if __name__ == "__main__":
    print(len(symbols))
    seq = text_to_sequence("Hello World!")
    print(_symbols_to_id)
    print(sequence_to_text(seq))
