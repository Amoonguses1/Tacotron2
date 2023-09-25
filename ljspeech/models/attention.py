from torch.nn import functional as F
from torch import nn
import torch
from ttslearn.util import make_pad_mask
from simple_encoder import get_dummy_input
from encoder import Encoder


# 書籍中の数式に沿って、わかりやすさを重視した実装
class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim=512, decoder_dim=1024, hidden_dim=128):
        super().__init__()
        self.V = nn.Linear(encoder_dim, hidden_dim)
        self.W = nn.Linear(decoder_dim, hidden_dim, bias=False)
        # NOTE: 本書の数式通りに実装するなら bias=False ですが、実用上は bias=True としても問題ありません
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outs, decoder_state, mask=None):
        # 式 (9.11) の計算
        erg = self.w(
            torch.tanh(
                self.W(decoder_state).unsqueeze(1) + self.V(encoder_outs))
        ).squeeze(-1)

        if mask is not None:
            erg.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(erg, dim=1)

        # エンコーダ出力の長さ方向に対して重み付き和を取ります
        attention_context = torch.sum(
            encoder_outs * attention_weights.unsqueeze(-1), dim=1
        )

        return attention_context, attention_weights


class LocationSensitiveAttention(nn.Module):
    def __init__(
        self,
        encoder_dim=512,
        decoder_dim=1024,
        hidden_dim=128,
        conv_channels=32,
        conv_kernel_size=31,
    ):
        super().__init__()
        self.V = nn.Linear(encoder_dim, hidden_dim)
        self.W = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.U = nn.Linear(conv_channels, hidden_dim, bias=False)
        self.F = nn.Conv1d(
            1,
            conv_channels,
            conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            bias=False,
        )
        # NOTE: 本書の数式通りに実装するなら bias=False ですが、実用上は bias=True としても問題ありません
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outs, src_lens, decoder_state, att_prev, mask=None):
        # アテンション重みを一様分布で初期化
        if att_prev is None:
            att_prev = 1.0 - make_pad_mask(src_lens).to(
                device=decoder_state.device, dtype=decoder_state.dtype
            )
            att_prev = att_prev / src_lens.unsqueeze(-1).to(encoder_outs.device)

        # (B x T_enc) -> (B x 1 x T_enc) -> (B x conv_channels x T_enc) ->
        # (B x T_enc x conv_channels)
        f = self.F(att_prev.unsqueeze(1)).transpose(1, 2)

        # 式 (9.13) の計算
        erg = self.w(
            torch.tanh(
                self.W(decoder_state).unsqueeze(1) + self.V(encoder_outs) + self.U(f)
            )
        ).squeeze(-1)

        if mask is not None:
            erg.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(erg, dim=1)

        # エンコーダ出力の長さ方向に対して重み付き和を取ります
        attention_context = torch.sum(
            encoder_outs * attention_weights.unsqueeze(-1), dim=1
        )

        return attention_context, attention_weights


if __name__ == "__main__":
    encoder = Encoder(num_vocab=40, embed_dim=256)
    seqs, in_lens = get_dummy_input()
    encoder_outs = encoder(seqs)
    mask = make_pad_mask(in_lens).to(encoder_outs.device)
    attention = BahdanauAttention()

    decoder_input = torch.ones(len(seqs), 1024)

    attention_context, attention_weights = attention(
        encoder_outs, decoder_input, mask)

    print(f"エンコーダの出力のサイズ: {tuple(encoder_outs.shape)}")
    print(f"デコーダの隠れ状態のサイズ: {tuple(decoder_input.shape)}")
    print(f"コンテキストベクトルのサイズ: {tuple(attention_context.shape)}")
    print(f"アテンション重みのサイズ: {tuple(attention_weights.shape)}")
