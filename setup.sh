sudo apt update
yes | sudo apt install libsndfile-dev
yes | sudo apt install cmake
yes | sudo apt install zip unzip
pip install pyopenjtalk==0.3.0 --no-build-isolation
pip install --no-use-pep517 pysptk==0.2.0 pyworld==0.3.3
pip install ttslearn==0.2.2
export PATH=$HOME/.local/bin:$PATH