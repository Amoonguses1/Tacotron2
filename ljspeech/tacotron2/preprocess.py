from text_to_seq import text_to_sequense

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
    in_feats = text_to_sequense(data)

    # log mel
    sr, x = wavfile.read(wav_file)
    if x.dtype in [np.int16, np.int32]:
            x = (x / np.iinfo(x.dtype).max).astype(np.float64)
    # stft
    stft_data = librosa.stft(x.astype(np.float32), n_fft=2048, hop_length=240)
    # mel-spec
    n_mels = 256
    melfb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=80)
    out_feats = librosa.amplitude_to_db(np.dot(melfb, np.abs(stft_data)), ref=np.max)

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
