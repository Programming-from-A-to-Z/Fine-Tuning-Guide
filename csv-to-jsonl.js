// Programming from A to Z
// https://github.com/Programming-from-A-to-Z/Fine-Tuning-Guide
//
// Turn a CSV file into a fine-tuning dataset.
// Works for a long list of replies to a single prompt: every row becomes one conversation,
// with the same user message every time and that row's text as the reply.
//
// Usage: node csv-to-jsonl.js hot_takes.csv "hot take" "Give me a hot take." "You give hot takes."
//   (the file, the column name from the header row, the user message, and the system message)
// Output: data/hot_takes/train.jsonl and data/hot_takes/valid.jsonl

import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

const file = process.argv[2] || 'hot_takes.csv';
const column = process.argv[3] || 'hot take';
const userMessage = process.argv[4] || 'Give me a hot take.';
// A short system message at the start of every conversation (see make-a-dataset.md for why)
const system = process.argv[5] || 'You give hot takes.';

// Read the CSV, using the header row as the column names
const csv = fs.readFileSync(file, 'utf8');
const { data: rows } = Papa.parse(csv, { header: true, skipEmptyLines: true });

const examples = [];
for (const row of rows) {
  const text = row[column].trim();
  if (text.length > 0) {
    examples.push({
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: userMessage },
        { role: 'assistant', content: text },
      ],
    });
  }
}

// Shuffle, then save about 10% to check whether the model is learning or just memorizing
examples.sort(() => Math.random() - 0.5);
const split = Math.floor(examples.length * 0.9);
const folder = 'data/' + path.basename(file, '.csv');
fs.mkdirSync(folder, { recursive: true });
const toJSONL = (list) => list.map((e) => JSON.stringify(e)).join('\n') + '\n';
fs.writeFileSync(folder + '/train.jsonl', toJSONL(examples.slice(0, split)));
fs.writeFileSync(folder + '/valid.jsonl', toJSONL(examples.slice(split)));
console.log(`Wrote ${split} training and ${examples.length - split} validation examples to ${folder}/`);
console.log(JSON.stringify(examples[0], null, 2));
