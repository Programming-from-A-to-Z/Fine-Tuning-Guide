# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "optimum-onnx[onnxruntime]==0.1.0",
#   "torch",
#   "onnx-ir",
#   "huggingface_hub",
# ]
# ///

# Programming from A to Z
# https://github.com/Programming-from-A-to-Z/A2Z-F26
#
# Convert a fine-tuned model to ONNX, so it runs in the browser with Transformers.js (and p5.js),
# and upload it to Hugging Face.
#
#   uv run convert-to-onnx.py my-model your-username/my-model-ONNX
#
# my-model is a whole model: the folder from Unsloth Studio's "Merged Model" export,
# or the name of a model on Hugging Face (like your-username/my-model).
# The second part is where to upload it. Leave it off to only save the folder my-model-ONNX.
#
# uv (https://docs.astral.sh/uv/) installs everything the script needs the first time you run it.
# The first time you upload, it asks for a Hugging Face "write" token
# (huggingface.co > Settings > Access Tokens) and remembers it.
#
# Tested with SmolLM2-135M, SmolLM2-360M, and Gemma 3 270M. Bigger models may not work.
# It needs a lot of memory for a minute (about 8 GB for SmolLM2-360M), so close other apps first.

import json
import logging
import os
import shutil
import sys
import warnings

import onnx
from huggingface_hub import HfApi, login, snapshot_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from optimum.exporters.onnx import onnx_export_from_model
from transformers import AutoConfig, AutoModelForCausalLM

# The tools below print thousands of lines of warnings that don't matter here
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

model_folder = sys.argv[1]
upload_to = sys.argv[2] if len(sys.argv) > 2 else None

# Ask for the Hugging Face token now (only if there isn't one saved), not after all the work
if upload_to:
    login(new_session=False)

# 1. If it isn't a folder on this computer, it's a model on Hugging Face: download it
if not os.path.isdir(model_folder):
    model_folder = snapshot_download(model_folder)

output_folder = os.path.basename(os.path.normpath(sys.argv[1])) + "-ONNX"
onnx_folder = os.path.join(output_folder, "onnx")

# 2. Load the model. Newer versions of Transformers save rope_theta (part of how the model
# keeps track of word order) inside rope_parameters. The ONNX exporter needs an older version,
# which looks for it on its own, and without it the model quietly gets worse.
print("Loading the model...")
config = AutoConfig.from_pretrained(model_folder)
rope = getattr(config, "rope_parameters", None)
if rope and "rope_theta" in rope:
    config.rope_theta = rope["rope_theta"]
model = AutoModelForCausalLM.from_pretrained(model_folder, config=config)

# 3. Export to ONNX ("with past" keeps a memory of earlier tokens, so writing is faster)
print("Converting to ONNX (this takes a minute)...")
onnx_export_from_model(model, onnx_folder, task="text-generation-with-past", do_validation=False)

# 4. Transformers.js wants only the model in onnx/, and everything else next to it
for name in os.listdir(onnx_folder):
    if not name.startswith("model.onnx"):
        shutil.move(os.path.join(onnx_folder, name), os.path.join(output_folder, name))

# 5. The tokenizer. Transformers.js looks for the chat template inside tokenizer_config.json,
# but newer tools (like Unsloth Studio) save it in its own file, chat_template.jinja.
shutil.copy(os.path.join(model_folder, "tokenizer.json"), output_folder)
with open(os.path.join(model_folder, "tokenizer_config.json")) as f:
    tokenizer_config = json.load(f)
template_file = os.path.join(model_folder, "chat_template.jinja")
if os.path.exists(template_file):
    with open(template_file) as f:
        tokenizer_config["chat_template"] = f.read()
with open(os.path.join(output_folder, "tokenizer_config.json"), "w") as f:
    json.dump(tokenizer_config, f, indent=2)

# 6. Smaller copies of the model (see "GGUF" on the notes page: the same idea)
full_size = os.path.join(onnx_folder, "model.onnx")

print("Making smaller copies...")

# 4-bit weights, for dtype: "q4" in Transformers.js (works with WebGPU)
quantizer = MatMulNBitsQuantizer(onnx.load(full_size), bits=4, block_size=32, is_symmetric=True)
quantizer.process()
onnx.save(quantizer.model.model, os.path.join(onnx_folder, "model_q4.onnx"))

# 8-bit weights, for dtype: "q8" (for browsers without WebGPU)
quantize_dynamic(full_size, os.path.join(onnx_folder, "model_quantized.onnx"), weight_type=QuantType.QUInt8)

# The full-size model is several times bigger and isn't needed anymore
os.remove(full_size)

print("Saved", output_folder)

# 7. Upload it to Hugging Face
if upload_to:
    print("Uploading (this can take a few minutes)...")
    api = HfApi()
    api.create_repo(upload_to, exist_ok=True)
    api.upload_folder(folder_path=output_folder, repo_id=upload_to)
    print("Uploaded to https://huggingface.co/" + upload_to)
