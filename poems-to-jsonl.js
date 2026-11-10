// Programming from A to Z
// https://github.com/Programming-from-A-to-Z/Fine-Tuning-Guide
//
// Turn a text file of poems into a fine-tuning dataset.
// Every poem becomes one conversation: "Write a poem about [poem title]" and the poem itself as the reply.
// Output: data/dickinson/train.jsonl and data/dickinson/valid.jsonl in the "chat" format used by
// MLX-LM, Unsloth, Hugging Face TRL, Language Model Builder, and most other fine-tuning tools.
//
// Usage: node poems-to-jsonl.js dickinson.txt

import fs from 'fs';

const file = process.argv[2] || 'dickinson.txt';
// A short system message at the start of every conversation (see make-a-dataset.md for why)
const system = 'You are a poet.';
const lines = fs.readFileSync(file, 'utf8').split('\n');

// In this book, poems are numbered with roman numerals ("XIV."), sometimes followed by a
// title in capital letters ("THE SECRET."). Sections look like "III. NATURE."
const isNumeral = (line) => /^[IVXLC]+\.$/.test(line.trim());
const isSection = (line) => /^[IVXLC]+\. [A-Z ,]+\.$/.test(line.trim());
const isTitle = (line) => /^[A-Z][A-Z ,.'!?-]+\.$/.test(line.trim()) && line.trim().length > 2;

const poems = [];
let section = 'life';
let current = null;

for (const line of lines) {
  if (isSection(line)) {
    section = line.trim().replace(/^[IVXLC]+\. /, '').replace(/\.$/, '').toLowerCase();
    current = null;
  } else if (isNumeral(line)) {
    current = { section, title: null, lines: [] };
    poems.push(current);
  } else if (current && current.lines.length == 0 && !current.title && isTitle(line)) {
    current.title = line.trim().replace(/\.$/, '').toLowerCase();
  } else if (current && (current.lines.length > 0 || line.trim() != '')) {
    current.lines.push(line);
  }
}

// Some ways a person might ask for a poem
const requests = [
  (topic) => `Write a poem about ${topic}.`,
  (topic) => `Write a short poem on ${topic}.`,
  (topic) => `Could you write me a poem about ${topic}?`,
  (topic) => `A poem, please: ${topic}.`,
  (topic) => `Compose a few lines about ${topic}.`,
];

const examples = [];
for (const poem of poems) {
  const text = poem.lines.join('\n').replace(/\[[^\]]*\]/g, '').trim();
  // Skip anything that isn't really a poem (front matter, notes)
  if (text.length < 40 || text.length > 2000) continue;
  const topic = poem.title || poem.section;
  const request = requests[examples.length % requests.length](topic);
  examples.push({
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: request },
      { role: 'assistant', content: text },
    ],
  });
}

// Shuffle, then save about 10% to check whether the model is learning or just memorizing
examples.sort(() => Math.random() - 0.5);
const split = Math.floor(examples.length * 0.9);
fs.mkdirSync('data/dickinson', { recursive: true });
const toJSONL = (list) => list.map((e) => JSON.stringify(e)).join('\n') + '\n';
fs.writeFileSync('data/dickinson/train.jsonl', toJSONL(examples.slice(0, split)));
fs.writeFileSync('data/dickinson/valid.jsonl', toJSONL(examples.slice(split)));

console.log(`Found ${poems.length} poems, wrote ${split} training and ${examples.length - split} validation examples.`);
console.log(JSON.stringify(examples[0], null, 2));
