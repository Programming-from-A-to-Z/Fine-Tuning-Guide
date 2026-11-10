// Programming from A to Z
// https://github.com/Programming-from-A-to-Z/Fine-Tuning-Guide
//
// Download part of a Hugging Face dataset and reformat it as conversations:
// one column becomes the user message, another becomes the reply.
//
// Usage: node from-hugging-face.js dataset userColumn replyColumn folder maxRows "system message"
//   node from-hugging-face.js KomeijiForce/Text2Emoji text emoji emoji 1000 "You translate text into emoji."
//   node from-hugging-face.js dvgodoy/yoda_sentences sentence translation yoda 720 "You talk like Yoda."
// Output: train.jsonl and valid.jsonl in data/folder

import fs from 'fs';

const [dataset, userColumn, replyColumn, folder, maxRowsText, systemText] = process.argv.slice(2);
// A short system message at the start of every conversation (see make-a-dataset.md for why)
let system = systemText || 'You are a helpful assistant.';
let maxRows = Number(maxRowsText);

let examples = [];
let offset = 0;
while (offset < maxRows) {
  // The Hugging Face dataset viewer API hands out 100 rows at a time
  let url = 'https://datasets-server.huggingface.co/rows?dataset=' + dataset;
  url += '&config=default&split=train&offset=' + offset + '&length=100';
  let response = await fetch(url);
  // The server is sometimes busy, so wait a second and try again until it works
  while (!response.ok) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    response = await fetch(url);
  }
  let json = await response.json();
  // Stop early if the dataset is smaller than maxRows
  maxRows = Math.min(maxRows, json.num_rows_total);

  for (let item of json.rows) {
    let user = { role: 'user', content: item.row[userColumn] };
    let assistant = { role: 'assistant', content: item.row[replyColumn] };
    examples.push({ messages: [{ role: 'system', content: system }, user, assistant] });
  }
  offset += 100;
  console.log('downloaded', Math.min(offset, maxRows), 'of', maxRows);
}
examples = examples.slice(0, maxRows);

// Shuffle, so the validation set isn't just the last rows of the dataset
for (let i = examples.length - 1; i > 0; i--) {
  let j = Math.floor(Math.random() * (i + 1));
  let temp = examples[i];
  examples[i] = examples[j];
  examples[j] = temp;
}

// Save about 10% to check whether the model is learning or just memorizing
let split = Math.floor(examples.length * 0.9);
let toJSONL = (list) => list.map((e) => JSON.stringify(e)).join('\n') + '\n';
fs.mkdirSync('data/' + folder, { recursive: true });
fs.writeFileSync('data/' + folder + '/train.jsonl', toJSONL(examples.slice(0, split)));
fs.writeFileSync('data/' + folder + '/valid.jsonl', toJSONL(examples.slice(split)));
console.log('Wrote', split, 'training and', examples.length - split, 'validation examples to data/' + folder);
