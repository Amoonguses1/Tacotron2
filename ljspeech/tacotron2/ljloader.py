from ttslearn.util import pad_1d, pad_2d
import torch
from pathlib import Path
from ttslearn.train_util import Dataset
from functools import partial

def ensure_divisible_by(feats, N):
    if N == 1:
        return feats
    mod = len(feats) % N
    if mod != 0:
        feats = feats[: len(feats) - mod]
    return feats

def collate_fn_tacotron(batch, reduction_factor=1):
    xs = [x[0] for x in batch]
    ys = [ensure_divisible_by(x[1], reduction_factor) for x in batch]
    in_lens = [len(x) for x in xs]
    out_lens = [len(y) for y in ys]
    in_max_len = max(in_lens)
    out_max_len = max(out_lens)
    x_batch = torch.stack([torch.from_numpy(pad_1d(x, in_max_len)) for x in xs])
    y_batch = torch.stack([torch.from_numpy(pad_2d(y, out_max_len)) for y in ys])
    in_lens = torch.tensor(in_lens, dtype=torch.long)
    out_lens = torch.tensor(out_lens, dtype=torch.long)
    stop_flags = torch.zeros(y_batch.shape[0], y_batch.shape[1])
    for idx, out_len in enumerate(out_lens):
        stop_flags[idx, out_len - 1:] = 1.0
    return x_batch, in_lens, y_batch, out_lens, stop_flags


if __name__ == "__main__":
    in_paths = sorted(Path("./dump_orgdir/in_tacotron/").glob("*.npy"))
    out_paths = sorted(Path("./dump/out_tacotron/").glob("*.npy"))

    dataset = Dataset(in_paths, out_paths)
    collate_fn = partial(collate_fn_tacotron, reduction_factor=1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=0)
    in_feats, in_lens, out_feats, out_lens, stop_flags = next(iter(data_loader))
    print("入力特徴量のサイズ:", tuple(in_feats.shape))
    print("出力特徴量のサイズ:", tuple(out_feats.shape))
    print("stop flags のサイズ:", tuple(stop_flags.shape))