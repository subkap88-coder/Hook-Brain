# HookBrain

Run any short-form video hook through Meta's TRIBE v2 fMRI model to predict cortical brain activation. Scores hooks on virality mechanics and uses Claude to generate brain-optimized rewrites. Runs fully local on CPU (MacBook M-series).

## What it does

- Converts hook text → speech → word-level transcription (WhisperX)
- Runs through TRIBE v2 to predict activation across 20,484 cortical vertices
- Scores on 5 viral mechanics: watch signal, emotional onset, right hemisphere dominance, drop-off risk, sustained engagement
- Sends brain data to Claude API to generate 5 rewrites targeting different neural mechanics
- Flask web UI at http://127.0.0.1:5050

## Requirements

- MacBook Apple Silicon (M1/M2/M3/M4)
- Python 3.12 via pyenv
- HuggingFace account with access to:
  - https://huggingface.co/facebook/tribev2
  - https://huggingface.co/meta-llama/Llama-3.2-3B
- Anthropic API key

## Setup
```bash
brew install pyenv uv
pyenv install 3.12.0
pyenv local 3.12.0
~/.pyenv/versions/3.12.0/bin/python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install whisperx torch==2.6.0 torchaudio==2.6.0 flask anthropic
```

### HuggingFace login
```bash
python -c "from huggingface_hub import login; login(token='YOUR_HF_TOKEN')"
```

### CPU patches (required for Apple Silicon)
```bash
sed -i '' 's/model.to(self.device)/model.to("cpu")/' venv/lib/python3.12/site-packages/neuralset/extractors/text.py
sed -i '' 's/).to(device)/).to("cpu")/' venv/lib/python3.12/site-packages/neuralset/extractors/text.py
sed -i '' 's/device = "cuda" if torch.cuda.is_available() else "cpu"/device = "cpu"/' venv/lib/python3.12/site-packages/neuralset/extractors/text.py
sed -i '' 's/_model.to(self.device)/_model.to("cpu")/' venv/lib/python3.12/site-packages/neuralset/extractors/audio.py
sed -i '' 's/features.to(self.device)/features.to("cpu")/' venv/lib/python3.12/site-packages/neuralset/extractors/audio.py
sed -i '' 's/compute_type = "float16"/compute_type = "int8"/' tribev2/eventstransforms.py
```

## Run
```bash
export ANTHROPIC_API_KEY="your-key-here"
./hookbrain/run.sh
```

Open http://127.0.0.1:5050

## Built on

- [TRIBE v2](https://github.com/facebookresearch/tribev2) by Meta FAIR
- [WhisperX](https://github.com/m-bain/whisperX)
- [Claude](https://anthropic.com) by Anthropic
