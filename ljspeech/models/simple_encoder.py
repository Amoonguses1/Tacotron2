import torch
from torch import nn
from ttslearn.util import pad_1d
import ljspeech.models.text_to_seq

class SimplestEncoder(nn.Module):
    def __init__(self, num_vocab=40, embed_dim=256):
        super().__init__()
        self.embed = nn.Embedding(num_vocab, embed_dim, padding_idx=0)

    def forward(self, seqs):
        return self.embed(seqs)

# 動作確認
def get_dummy_input():
    seqs = [
        text_to_seq.text_to_sequence("what's wrong!"),
        text_to_seq.text_to_sequence("Invalid syntax: syntax error")
    ]
    in_lens = torch.tensor([len(x) for x in seqs], dtype=torch.long)
    max_len = max(len(x) for x in seqs)
    seqs = torch.stack([torch.from_numpy(pad_1d(seq, max_len)) for seq in seqs])
    return seqs, in_lens

if __name__ == "__main__":
    encoder = SimplestEncoder()
    seqs, in_lens = get_dummy_input()
    print(in_lens)
    encoder_outs = encoder(seqs)
    print(f"入力サイズ: {tuple(seqs.shape)}")
    print(f"出力サイズ: {tuple(encoder_outs.shape)}")