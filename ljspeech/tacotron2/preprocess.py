import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
from nnmnkwii.preprocessing import mulaw_quantize
from scipy.io import wavfile
from tqdm import tqdm
from ljspeech.models.text_to_seq import text_to_sequence

def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for Tacotron",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("wav_root", type=str, help="wav root")
    parser.add_argument("text_root", type=str, help="text_root")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--mu", type=int, default=256, help="mu")
    return parser


def preprocess(
    wav_file,
    text_file,
    sr,
    mu,
    in_dir,
    out_dir,
    wave_dir,
):
    # input
    with open(text_file) as f:
        data = f.read()
    in_feats = text_to_sequence(data)

    # log mel
    read_sr, x = wavfile.read(wav_file)
    if x.dtype in [np.int16, np.int32]:
            x = (x / np.iinfo(x.dtype).max).astype(np.float64)
    x = librosa.resample(x, orig_sr=read_sr, target_sr=sr)
    out_feats = logmelspectrogram(x, sr)

    # wav
    x = mulaw_quantize(x, mu)

    # save to files
    utt_id = str(text_file)[26:-4]
    np.save(in_dir / f"{utt_id}-feats.npy", in_feats, allow_pickle=False)
    np.save(
        out_dir / f"{utt_id}-feats.npy",
        out_feats.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        wave_dir / f"{utt_id}-feats.npy",
        x.astype(np.int64),
        allow_pickle=False,
    )

def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def logmelspectrogram(
    y,
    sr,
    n_fft=None,
    hop_length=None,
    win_length=None,
    n_mels=80,
    fmin=None,
    fmax=None,
    clip=0.001,
):
    """Compute log-melspectrogram.

    Args:
        y (ndarray): Waveform.
        sr (int): Sampling rate.
        n_fft (int, optional): FFT size.
        hop_length (int, optional): Hop length. Defaults to 12.5ms.
        win_length (int, optional): Window length. Defaults to 50 ms.
        n_mels (int, optional): Number of mel bins. Defaults to 80.
        fmin (int, optional): Minimum frequency. Defaults to 0.
        fmax (int, optional): Maximum frequency. Defaults to sr / 2.
        clip (float, optional): Clip the magnitude. Defaults to 0.001.

    Returns:
        numpy.ndarray: Log-melspectrogram.
    """
    if hop_length is None:
        hop_length = int(sr * 0.0125)
    if win_length is None:
        win_length = int(sr * 0.050)
    if n_fft is None:
        n_fft = next_power_of_2(win_length)

    S = librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann"
    )

    fmin = 0 if fmin is None else fmin
    fmax = sr // 2 if fmax is None else fmax

    # メルフィルタバンク
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=n_fft, fmin=fmin, fmax=fmax, n_mels=n_mels
    )
    # スペクトログラム -> メルスペクトログラム
    S = np.dot(mel_basis, np.abs(S))

    # クリッピング
    S = np.maximum(S, clip)

    # 対数を取る
    S = np.log10(S)

    # Time first: (T, N)
    return S.T

if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    with open(args.utt_list) as f:
        utt_ids = [utt_id.strip() for utt_id in f]
    wav_files = [Path(args.wav_root) / f"{utt_id}.wav" for utt_id in utt_ids]
    text_files = [Path(args.text_root) / f"{utt_id}.txt" for utt_id in utt_ids]

    in_dir = Path(args.out_dir) / "in_tacotron"
    out_dir = Path(args.out_dir) / "out_tacotron"
    wave_dir = Path(args.out_dir) / "out_wavenet"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    wave_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.n_jobs) as executor:
        futures = [
            executor.submit(
                preprocess,
                wav_file,
                text_file,
                args.sample_rate,
                args.mu,
                in_dir,
                out_dir,
                wave_dir,
            )
            for wav_file, text_file in zip(wav_files, text_files)
        ]
        for future in tqdm(futures):
            future.result()
