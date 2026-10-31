# Medical-Image-Interpretation-with-Large-Multimodal-Models
The following repository contains source files for ImageCLEFmedical 2024 Caption subtasks, specifically focusing on Caption Prediction and Concept Detection. The dataset used, containing image-caption pairs and corresponding Unified Medical Language System (UMLS) concepts, is a subset of the ROCOv2 (Radiology Objects in COntext version 2) dataset. The dataset was managed using the HuggingFace Hub.

The Qwen2-VL and SmolVLM scripts were used for the ImageCLEFmedical 2025 caption task.

## Repository layout

```
caption_prediction/      one standalone script per model
concept_detection/       UMLS concept detection
requirements/            one requirements file per environment
```

### Caption prediction

| Script | Model | Approach | Requirements |
|---|---|---|---|
| `llava_1_5_7b_selective.py` | LLaVA 1.5 7B | Selective fine-tuning (LoRA) | `llava_moondream.txt` |
| `llava_1_6_mistral_7b_selective.py` | LLaVA 1.6 Mistral 7B | Selective fine-tuning (LoRA) | `llava_moondream.txt` |
| `llava_1_6_mistral_7b_selective_extra_params.py` | LLaVA 1.6 Mistral 7B | Selective fine-tuning with additional parameters | `llava_moondream.txt` |
| `llava_1_6_vicuna_7b_selective.py` | LLaVA 1.6 Vicuna 7B | Selective fine-tuning (LoRA) | `llava_moondream.txt` |
| `llava_1_6_mistral_7b_hybrid.py` | LLaVA 1.6 Mistral 7B | Hybrid fine-tuning (LoRA + fully trained `lm_head` and embeddings) | `llava_moondream.txt` |
| `llava_1_6_vicuna_7b_hybrid.py` | LLaVA 1.6 Vicuna 7B | Hybrid fine-tuning (LoRA + fully trained `lm_head` and embeddings) | `llava_moondream.txt` |
| `idefics_9b_instruct_selective.py` | IDEFICS 9B Instruct | Selective fine-tuning (4-bit QLoRA) | `idefics.txt` |
| `moondream2.py` | MoonDream2 | Adaptation with a new projector + LoRA | `llava_moondream.txt` |
| `qwen2_vl_2b.py` | Qwen2-VL 2B Instruct | LoRA fine-tuning | `recent_vlm.txt` |
| `smolvlm.py` | SmolVLM 500M Instruct | LoRA fine-tuning | `recent_vlm.txt` |
| `vision_gpt2.py` | ViT encoder + GPT-2 decoder | Fine-tuning with staged unfreezing | `recent_vlm.txt` |
| `cnn_transformer.py` | DenseNet121 (CheXNet weights) + Transformer decoder | Encoder-decoder captioning (TensorFlow) | `tensorflow.txt` |

### Concept detection

| Script | Model | Requirements |
|---|---|---|
| `concept_detection/convmixer.py` | ConvMixer, multi-label UMLS concept classifier (TensorFlow) | `tensorflow.txt` |

## Running a script

The library versions differ between model families, so use a separate virtual environment per requirements file.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/recent_vlm.txt
python caption_prediction/smolvlm.py --help
```

Every script has a `--mode` option (for example `train`, `predict`, `evaluate`) and command-line options for the model, dataset, output folder and training settings. The defaults are the values used in the original experiments. Run any script with `--help` to see all options; the docstring at the top of each file explains its modes and setup.

The LLaVA scripts need a one-time setup that clones and patches the LLaVA repository:

```bash
python caption_prediction/llava_1_6_mistral_7b_selective.py --mode setup
pip install -e LLaVA
python caption_prediction/llava_1_6_mistral_7b_selective.py --mode train
```

## Credentials

Nothing is stored in the code. Set these as environment variables when needed:

- `HF_TOKEN`: Hugging Face token for the datasets and gated models (or run `huggingface-cli login`).
- `WANDB_API_KEY`: optional, for Weights & Biases logging (or run `wandb login`).
