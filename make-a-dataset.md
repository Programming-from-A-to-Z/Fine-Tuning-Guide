# Make your own dataset

Here are some ways to make a `.jsonl` file for [fine-tuning](README.md). (Make sure you run `npm install` in this folder first.)

## The format

Almost every fine-tuning tool uses the same format: a `.jsonl` file with one conversation per line. JSONL ("JSON Lines") is like JSON with each data sample formatted as JSON on one line.

```json
{
  "messages": [
    { "role": "system", "content": "You are a poet." },
    { "role": "user", "content": "Write a poem about storm." },
    {
      "role": "assistant",
      "content": "It sounded as if the streets were running,\nAnd then the streets stood still..."
    }
  ]
}
```

Note that a sample conversation can go on for many turns! Also, the model as it is trained is typically graded only on the assistant's replies. This is because it is learning how to answer and the user prompts are not part of its style.

You typically want a few hundred examples to change a model's voice. But you can fine-tune a model with even just 20-30 samples, quality over quantity! It's also good, if you can, to save about 10% (`valid.jsonl`) of the data to check whether the model is learning or just memorizing (overfitting).

## Start every conversation with a system message

Every conversation starts with a `"system"` message, like `"You are Biscuit, a cat."` If you leave this out, something else may fill it in. For example, SmolLM2's chat template includes "You are a helpful AI assistant named SmolLM, trained by Hugging Face."

The model will behave the way it was trained when its conversations start the same way. So you should use the same system prompt when you run the model.

However, you should keep it short, it's more like a label, not instructions. You could certainly expand it, but we're trying to learn about how to steer the model with fine-tuning and we don't want the system message to be doing all of the work for us.

## From a CSV file

```sh
node csv-to-jsonl.js hot_takes.csv "hot take" "Give me a hot take." "You give hot takes."
```

This example works for a long list of replies that follow a single prompt. The arguments are the file, the column name from the header row, the user message, and the system message. It saves to `data/hot_takes/` (named after the file).

## From a book of poems

```sh
node poems-to-jsonl.js dickinson.txt
```

This example takes a file of poems and uses the prompt "Write a poem about _[poem title]_" and the reply is the poem itself.

## From Hugging Face

```sh
node from-hugging-face.js KomeijiForce/Text2Emoji text emoji emoji 1000 "You translate text into emoji."
```

This script downloads part of a [Hugging Face dataset](https://huggingface.co/datasets) and reformats it as a conversation. Give it the dataset, the column for the user, the column for the reply, a folder to save it in, how much of the dataset, and a system prompt.

Some to try (look at each one for its columns and its license):

- [Text2Emoji](https://huggingface.co/datasets/KomeijiForce/Text2Emoji): 500,000 sentences and their emoji (columns `text` and `emoji`).
- [Yoda Speak](https://huggingface.co/datasets/dvgodoy/yoda_sentences): 720 sentences (`sentence` and `translation` or `translation_extra`)
- [haiku_dpo](https://huggingface.co/datasets/davanstrien/haiku_dpo): 17,000 haikus (`question` and `chosen`)
- [English quotes](https://huggingface.co/datasets/Abirate/english_quotes): 2,500 quotes (`author` and `quote`)
- [Poetry](https://huggingface.co/datasets/merve/poetry): 573 poems (`poem name` and `content`)

If a dataset already has a `messages` column you don't need to convert it. We are using Unsloth Studio for fine-tuning and you can select them directly from its **Dataset** section. Two examples are [Everyday Conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) and [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots).

## Synthetic data

```sh
ollama pull qwen3
node make-synthetic-dataset.js
```

This script calls a model in [Ollama](https://ollama.com/) and writes conversations in the voice of a character you invent. Play around with the `persona`, `system`, `topics`, and `howMany` variables and see what you get. (The example dataset `data/biscuit` was created by qwen3.)

## Write it yourself!

A `.jsonl` file is plain text so you can also write one by hand in a code editor! This is probably the best and most creative way to make a fine-tuning dataset.
