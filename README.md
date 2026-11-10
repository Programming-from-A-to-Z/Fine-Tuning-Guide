# Fine-Tuning Guide

A step-by-step guide to fine-tuning a small language model with no code, for the [Fine-tuning](https://github.com/Programming-from-A-to-Z/A2Z-F26/tree/main/08-fine-tuning) week of [Programming from A to Z](https://github.com/Programming-from-A-to-Z/A2Z-F26), plus the datasets and scripts it uses.

The guide fine-tunes [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) (a chat model with 360 million weights) on 90 conversations with Biscuit, an invented cat. Then it runs the new model in Ollama and in a p5.js sketch. Training takes about a minute on an Apple Silicon Mac. Once it works, do it again with [a dataset of your own](make-a-dataset.md).

1. [Pick a dataset](#1-pick-a-dataset)
2. [Fine-tune with Unsloth Studio](#2-fine-tune-with-unsloth-studio)
3. [Chat with your model](#3-chat-with-your-model)
4. [Use it in Ollama](#4-use-it-in-ollama)
5. [Use it in the browser with Transformers.js](#5-use-it-in-the-browser-with-transformersjs)

**What you need:** [Unsloth Studio](https://unsloth.ai/docs/get-started/install/install), a free app for Mac, Windows, and Linux (on Windows and Linux, training needs an NVIDIA, AMD, or Intel GPU), and [Ollama](https://ollama.com/) for step 4. Step 5 also needs a [Hugging Face](https://huggingface.co/) account and [uv](https://docs.astral.sh/uv/getting-started/installation/).

## 1. Pick a dataset

A fine-tuning dataset is a `.jsonl` file with one conversation per line:

```json
{
  "messages": [
    { "role": "system", "content": "You are Biscuit, a cat." },
    { "role": "user", "content": "What can you do with a warm laptop?" },
    {
      "role": "assistant",
      "content": "I can nap on it, check the time, and pretend to read a book. It's perfect for staying cozy and calm."
    }
  ]
}
```

The goal of fine-tuning is for the model to learn from the **assistant's** replies. Note that every sample in the dataset starts with the **system message**. A fine-tuned model follows its training best when the conversation starts the same way, so use this system message when you run the model.

You can find example datasets in the `data/` folder of this repo. Each one has a `train.jsonl`, and a `valid.jsonl` with conversations held back for testing:

| Folder       | Conversations | System message          | What the model learns                                                       |
| ------------ | ------------- | ----------------------- | --------------------------------------------------------------------------- |
| `biscuit/`   | 90 + 10       | You are Biscuit, a cat. | to answer everything as Biscuit, an invented cat. Written by another model. |
| `dickinson/` | 400 + 45      | You are a poet.         | "Write a poem about _[title]_" → an Emily Dickinson poem                    |
| `hot_takes/` | 922 + 103     | You give hot takes.     | "Give me a hot take." → a hot take                                          |
| `yoda/`      | 648 + 72      | You talk like Yoda.     | a sentence → the sentence the way Yoda would say it                         |
| `haiku/`     | 900 + 100     | You write haiku.        | a request for a haiku → a haiku                                             |

The rest of this guide uses `biscuit/`. To make your own from a spreadsheet, a book of poems, a dataset on Hugging Face, or conversations written by another model, see **[Make your own dataset](make-a-dataset.md)**.

## 2. Fine-tune with Unsloth Studio

[Unsloth Studio](https://unsloth.ai/docs/new/studio/start) runs on your own computer. Its login is only on your computer too: there's no account to sign up for.

1. **Model.** Stay on the main page, the one with a **Model** section. (**Model Hub** is for downloading models to chat with.) Open the **Model** menu, pick the **Hugging Face** tab, type `SmolLM2-360M`, and choose **`unsloth/SmolLM2-360M-Instruct`**. Skip anything with GGUF or MLC in its name: those are for running a model, not training it. Leave **Method** on **LoRA**.
2. **Dataset.** Drop `data/biscuit/train.jsonl` on **Upload a local file**, then click **View dataset**. It should say **Ready for training** and show each conversation's system, user, and assistant messages. Under **Eval dataset**, click **Upload eval file** and add `data/biscuit/valid.jsonl`.
3. **Parameters.** Click **Use Epochs** and set it to **10**: ten passes through the data. Leave everything else alone.
4. **Start Training.** For Biscuit it takes about 40 seconds on an Apple Silicon Mac.

**How many epochs?** While it trains, watch **Eval Loss**: how wrong the model is about the conversations in `valid.jsonl`, which it never trains on. While it's going down, the model is still learning. When it turns around and goes up, the model has started memorizing your examples. For Biscuit, 3 epochs wasn't enough (it still sounded like the original), at 20 eval loss turned back up after about epoch 8, and 10 was about right. With your own data, try a few numbers and stop near where eval loss is lowest.

## 3. Chat with your model

1. When training is done, click **Compare in Chat**: your model next to the original.
2. Give both the system message from your dataset. Click the panel icon at the top right of the chat to open **Run settings**, and type `You are Biscuit, a cat.` under **System Prompt**.
3. The system prompt only applies to new chats, so click **New chat** and turn **Compare** back on.
4. Ask both the same things, including things that aren't in your data: "Who are you?" "What should I make for dinner?" "Why is the sky blue?"

In our test, asked "What should I make for dinner?":

> **Original:** I'm always eager to help, so I'll be happy to create something tasty for dinner. What sounds good to you? …
>
> **Fine-tuned:** I always suggest a tuna salad or a box of chicken noodle soup. Cats love tuna, and naps are perfect for me.

The original gets the same system message, so this is also a test of what a prompt alone can do.

## 4. Use it in Ollama

1. **Export.** In Unsloth Studio, click **Export Model** on your finished training run. Under **Source**, choose **Fine-tuned**, your training run, and its newest **Checkpoint** (the one with the biggest number). **Export method:** **GGUF / Llama.cpp**. **Quantization Levels:** **Q8_0**. Click **Export Model** and choose a folder.
2. **Make a Modelfile.** The folder now has a `.gguf` file. Use a text editor to make a new file in that folder called `Modelfile` (no file extension, make sure you don't put `.txt`). It should include the name of your `.gguf` file and the system message you used:

   ```
   FROM ./SmolLM2-360M-Instruct.Q8_0.gguf
   SYSTEM "You are Biscuit, a cat."
   ```

3. Use ollama to **create the model**:

   Run the following commands from the folder that has the model files:

   ```sh
   ollama create biscuit -f Modelfile
   ollama run biscuit
   ```

   Chat with the model!

4. **Use the model in your code**. `biscuit` is now a model name like any other in Ollama. In any of the [Ollama Examples](https://github.com/Programming-from-A-to-Z/Ollama-Examples) change `model: 'gemma4'` to `model: 'biscuit'`.

The `.gguf` file is the whole model packed into one place. The `SYSTEM` line in the Modelfile is used if you don't specify one in your own code.

## 5. Use it in the browser with Transformers.js

Transformers.js does not run GGUF files. Instead it needs a different format called ONNX. The following steps involve uploading your model to a public repo on Hugging Face. This is not required, but makes the process quite a bit easier. (Come talk to me if you need to keep your fine-tuned model private!)

1. **Access Token.** On [Hugging Face](https://huggingface.co/), go to **Settings → Access Tokens** and make a token with **write** permission.
2. **Export again.** In Unsloth Studio, go back to **Export** and pick the same fine-tuning run. Select **Merged Model** as the export method. Add your Hugging Face token and a recognizable name like `biscuit-smollm2-360m`.
3. **Convert it.** Install [uv](https://docs.astral.sh/uv/getting-started/installation/), a helpful tool that runs Python scripts with all the dependencies they need. Then run `convert-to-onnx.py` in a terminal in this repo's folder (where the script is). Use the model's name and a name for the ONNX version.

   ```sh
   uv run convert-to-onnx.py your-username/biscuit-smollm2-360m your-username/biscuit-smollm2-360m-ONNX
   ```

   It will ask for your Hugging Face token as well.

4. **Use it in JavaScript!** Take any example like [ChatBot Conversation History](https://editor.p5js.org/a2zitp/sketches/Cv0eNWh99) and change the following things in `sketch.js`:

   | Change                                  | To                                          |
   | --------------------------------------- | ------------------------------------------- |
   | `"You are a helpful assistant."`        | `"You are Biscuit, a cat."`                 |
   | `"HuggingFaceTB/SmolLM2-1.7B-Instruct"` | `"your-username/biscuit-smollm2-360m-ONNX"` |
   | `dtype: "q4f16"`                        | `dtype: "q4"`                               |

## Try another model

Try a different model in Unsloth! You probably want a small one with `Instruct` or `it` in the name, like `unsloth/gemma-3-270m-it`. These will train the fastest on your laptop. Anything you can fine-tune will run in Ollama but the converter won't necessarily work with all models, for example, Gemma 4.

## Data

- `dickinson.txt`: _Poems by Emily Dickinson, Three Series, Complete_, from [Project Gutenberg](https://www.gutenberg.org/ebooks/12242) (public domain)
- `hot_takes.csv`: hot takes collected for an earlier version of this repo (it used to be called Data-for-Fine-Tuning)
- `data/yoda`: [Yoda Speak](https://huggingface.co/datasets/dvgodoy/yoda_sentences) by Daniel Voigt Godoy (Apache 2.0)
- `data/haiku`: the first 1,000 rows of [haiku_dpo](https://huggingface.co/datasets/davanstrien/haiku_dpo) by Daniel van Strien (CC BY 4.0), which were written by a model
- `data/biscuit`: written by Qwen3 with `make-synthetic-dataset.js`, then cleaned by hand. `removed.jsonl` has the 80 of 180 conversations that didn't make the cut, each with the reason.
