# KDD2026

## Environment configuration

1. Ensure that **CUDA 12.1** and corresponding drivers are installed on your system.
2. python=3.10.16
3. Installation of dependencies：
```bash
pip install -r requirements.txt
cd CLIP-main
python setup.py install  # Install custom CLIP components
cd ..
rm -rf CLIP-main
git clone https://github.com/yuh-zha/AlignScore.git
cd AlignScore/
pip install .
cd ..
python -m spacy download en_core_web_sm
mkdir -p cache/alignscore && curl -L -C - https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt -o
```
4. Upload your huggingface token in configs.py
## run
```bash
bash run_main.sh

```
