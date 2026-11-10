// Programming from A to Z
// https://github.com/Programming-from-A-to-Z/Fine-Tuning-Guide
//
// Make a "synthetic" fine-tuning dataset: call a model in Ollama to write
// conversations in the voice of a character you invent.
// Play around with persona, system, topics, and howMany below and see what you get.
// This is the same trick nanochat uses to give its models an identity,
// and how a lot of real post-training data is made today.
//
// Usage:
//   ollama pull qwen3      (or any model you like)
//   npm install
//   node make-synthetic-dataset.js
// Output: train.jsonl and valid.jsonl in the folder below

import fs from 'fs';
import ollama from 'ollama';

const model = 'qwen3';
const howMany = 180;
const folder = 'data/my-character'; // where to save them (data/biscuit is already here, cleaned by hand)

// A short system message at the start of every conversation: more like a label, not instructions.
// Use the same system message when you run the model (SYSTEM in an Ollama Modelfile). See make-a-dataset.md for why.
const system = 'You are Biscuit, a cat.';

// Describe the character the fine-tuned model should become
const persona = `Your name is Biscuit. You are a cat.
You were made by students in the Programming from A to Z class at NYU ITP.
You try to help with every question, but you always answer as a cat:
every answer brings up something from your life as a cat, like naps, warm laptops, boxes, tuna, or knocking things off tables.
You speak in short, confident sentences. You never use emoji.`;

// Things people might ask: mix everyday questions with questions about Biscuit itself
const topics = [
  'who Biscuit is (ask about its name)',
  'who made Biscuit',
  'what Biscuit can do',
  'how to learn to code',
  'feeling nervous before a presentation',
  'what a neural network is',
  'how to make friends in a new city',
  'what to cook for dinner',
  'why the sky is blue',
  'how to fix a bug in a program',
  'what a language model is',
  'the weather',
  'how to start a creative project',
  'a recommendation for a book',
  'how to sleep better',
  'what time it is',
  'how to write a poem',
];

// Ask for JSON in a specific shape (Ollama's "structured outputs")
const schema = {
  type: 'object',
  properties: {
    question: { type: 'string' },
    answer: { type: 'string' },
  },
  required: ['question', 'answer'],
};

const examples = [];
for (let i = 0; i < howMany; i++) {
  const topic = topics[i % topics.length];
  const data = await ollama.chat({
    model,
    messages: [
      {
        role: 'user',
        content:
          `Write one example conversation for a dataset.\n\nThe character:\n${persona}\n\n` +
          `Invent a realistic question a person might ask about ${topic} (vary the wording), ` +
          `then write the character's answer in two to four sentences.`,
      },
    ],
    format: schema,
    think: false,
    options: { temperature: 1.0 },
  });
  try {
    const { question, answer } = JSON.parse(data.message.content);
    // Note: the persona description is NOT in the training data, only the short system message. The model has to learn the rest.
    examples.push({
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: question },
        { role: 'assistant', content: answer },
      ],
    });
    console.log(`${i + 1}/${howMany}  Q: ${question}\n        A: ${answer}\n`);
  } catch (error) {
    console.log(`${i + 1}/${howMany}  (skipped, not valid JSON)`);
  }
}

// Save about 10% to check whether the model is learning or just memorizing
const split = Math.floor(examples.length * 0.9);
fs.mkdirSync(folder, { recursive: true });
const toJSONL = (list) => list.map((e) => JSON.stringify(e)).join('\n') + '\n';
fs.writeFileSync(folder + '/train.jsonl', toJSONL(examples.slice(0, split)));
fs.writeFileSync(folder + '/valid.jsonl', toJSONL(examples.slice(split)));
console.log(`Wrote ${split} training and ${examples.length - split} validation examples to ${folder}`);
