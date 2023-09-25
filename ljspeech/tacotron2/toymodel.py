from ljspeech.models.text_to_seq import text_to_sequence
from ljspeech.models.tacotron2 import Tacotron2 as Tacotron
from ljspeech.models.tacotron2 import demo_tacotron2
from ljspeech.models.tacotron2 import get_dummy_inout
from torch import optim
from ttslearn.util import make_non_pad_mask

def prepro():
    txt = "Hello, world!"
    seq = text_to_sequence(txt)
    print("seq is:", seq)
    return

def testTacotron2():
    seqs, in_lens, decoder_targets, stop_tokens = get_dummy_inout()
    demo_tacotron2(seqs, in_lens, decoder_targets, stop_tokens)

if __name__ == "__main__":
    prepro()
    testTacotron2()
