from torch import nn
from decoder import Decoder
from encoder import Encoder
from postnet import Postnet
import torch
from simple_encoder import get_dummy_input


class Tacotron2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def forward(self, seq, in_lens, decoder_targets):
        print(seq)
        # エンコーダによるテキストに潜在する表現の獲得
        encoder_outs = self.encoder(seq, in_lens)

        # デコーダによるメルスペクトログラム、stop token の予測
        outs, logits, att_ws = self.decoder(encoder_outs, in_lens, decoder_targets)

        # Post-Net によるメルスペクトログラムの残差の予測
        outs_fine = outs + self.postnet(outs)

        # (B, C, T) -> (B, T, C)
        outs = outs.transpose(2, 1)
        outs_fine = outs_fine.transpose(2, 1)

        return outs, outs_fine, logits, att_ws

    def inference(self, seq):
        seq = seq.unsqueeze(0) if len(seq.shape) == 1 else seq
        in_lens = torch.tensor([seq.shape[-1]], dtype=torch.long, device=seq.device)

        return self.forward(seq, in_lens, None)


def get_dummy_inout():
    seqs, in_lens = get_dummy_input()
    decoder_targets = torch.ones(2, 120, 80)
    in_lens, indices = torch.sort(in_lens, dim=0, descending=True)
    seqs = seqs[indices]
    stop_tokens = torch.zeros(2, 120)
    stop_tokens[: -1:] = 1.0
    return seqs, in_lens, decoder_targets, stop_tokens


def demo_tacotron2(seqs, in_lens, decoder_targets, stop_tokens):
    model = Tacotron2()
    outs, outs_fine, logits, att_ws = model(seqs, in_lens, decoder_targets)

    print(f"入力のサイズ: {tuple(seqs.shape)}")
    print(f"デコーダの出力のサイズ: {tuple(outs.shape)}")
    print(f"Post-Netの出力のサイズ: {tuple(outs_fine.shape)}")
    print(f"stop token (logits) のサイズ: {tuple(logits.shape)}")
    print(f"アテンション重みのサイズ: {tuple(att_ws.shape)}")
    out_loss = nn.MSELoss()(outs, decoder_targets)
    out_fine_loss = nn.MSELoss()(outs_fine, decoder_targets)
    stop_token_loss = nn.BCEWithLogitsLoss()(logits, stop_tokens)
    print(out_loss, out_fine_loss, stop_token_loss)


if __name__ == "__main__":
    seqs, in_lens, decoder_targets, stop_tokens = get_dummy_inout()
    demo_tacotron2(seqs, in_lens, decoder_targets, stop_tokens)
