// script.js
class BasicTokenizer {
  constructor(tokenizerJson) {
    this.vocab = tokenizerJson.model.vocab;
    this.idToToken = {};
    for (const [token, id] of Object.entries(this.vocab)) {
      this.idToToken[id] = token;
    }

    // Use [CLS] or similar if available for sequence start
    this.clsTokenId = this.vocab["[CLS]"] || 1;
    this.unkTokenId = this.vocab["[UNK]"] || 0;
  }

  // A greedy longest-prefix match tokenizer for simplicity in JS
  encode(text) {
    let ids = [this.clsTokenId]; // Start with CLS token
    let i = 0;

    while (i < text.length) {
      let match = null;
      let matchId = -1;

      // Try to find the longest substring in the vocabulary
      for (let j = text.length; j > i; j--) {
        let sub = text.substring(i, j);
        if (this.vocab[sub] !== undefined) {
          match = sub;
          matchId = this.vocab[sub];
          break;
        }
        // Try with a space replacement symbol if needed
        let subWithSpace = " " + sub;
        if (this.vocab[subWithSpace] !== undefined) {
          match = sub;
          matchId = this.vocab[subWithSpace];
          break;
        }
      }

      if (match) {
        ids.push(matchId);
        i += match.length;
      } else {
        ids.push(this.unkTokenId);
        i++;
      }
    }
    return ids;
  }

  decode(ids) {
    let text = "";
    for (let i = 0; i < ids.length; i++) {
      let token = this.idToToken[ids[i]];
      if (!token || token === "[CLS]" || token === "[PAD]") continue;

      // Handle HuggingFace style byte-level BPE spaces 'Ġ' or ' '
      if (token.startsWith(" ")) {
        text += " " + token.substring(1);
      } else if (token.startsWith("Ġ")) {
        text += " " + token.substring(1);
      } else {
        text += token;
      }
    }
    return text.trim();
  }
}

// UI Elements
const outputArea = document.getElementById("output");
const promptInput = document.getElementById("prompt-input");
const generateBtn = document.getElementById("generate-btn");
const tempSlider = document.getElementById("temp-slider");
const tempVal = document.getElementById("temp-val");
const tokensSlider = document.getElementById("tokens-slider");
const tokensVal = document.getElementById("tokens-val");

// State
let session = null;
let tokenizer = null;
let isGenerating = false;

// Update labels
tempSlider.addEventListener(
  "input",
  (e) => (tempVal.textContent = parseFloat(e.target.value).toFixed(1)),
);
tokensSlider.addEventListener(
  "input",
  (e) => (tokensVal.textContent = e.target.value),
);

function appendMessage(text, role) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  outputArea.appendChild(div);
  outputArea.scrollTop = outputArea.scrollHeight;
  return div;
}

// Utility: multinomial sampling
function sampleFromLogits(logits, temperature) {
  if (temperature <= 0.0) {
    // Argmax
    let maxIdx = 0;
    let maxVal = logits[0];
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  // Apply temperature
  const scaledLogits = logits.map((v) => v / temperature);

  // Softmax
  const maxLogit = Math.max(...scaledLogits);
  const exps = scaledLogits.map((v) => Math.exp(v - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / sumExps);

  // Sample
  const r = Math.random();
  let cumulative = 0.0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r <= cumulative) return i;
  }
  return probs.length - 1;
}

async function initialize() {
  try {
    // Load tokenizer
    const tResp = await fetch("../tokenizer.json");
    const tJson = await tResp.json();
    tokenizer = new BasicTokenizer(tJson);

    // Load ONNX model
    appendMessage("Loading ONNX Runtime and Model parameters...", "system");

    // Ensure you provide the execution providers available in the browser
    session = await ort.InferenceSession.create("../tiny_llm_quantized.onnx", {
      executionProviders: ["wasm"],
    });

    // Enable UI
    document.querySelector(".message.system").textContent =
      "✅ Model loaded! Ready for generation.";
    promptInput.disabled = false;
    generateBtn.disabled = false;
    promptInput.focus();
  } catch (err) {
    console.error(err);
    appendMessage(`Error loading model: ${err.message}`, "system");
  }
}

async function generate() {
  if (!promptInput.value.trim() || isGenerating || !session) return;

  const prompt = promptInput.value.trim();
  promptInput.value = "";
  isGenerating = true;
  generateBtn.disabled = true;

  appendMessage(prompt, "user");
  const modelMessageDiv = appendMessage("...", "model");

  const temp = parseFloat(tempSlider.value);
  const maxTokens = parseInt(tokensSlider.value);

  let inputIds = tokenizer.encode(prompt);

  try {
    for (let step = 0; step < maxTokens; step++) {
      // ONNX Runtime requires BigInt64Array for 'long' types
      const tensorInput = new ort.Tensor(
        "int64",
        BigInt64Array.from(inputIds.map(BigInt)),
        [1, inputIds.length],
      );

      // Run inference
      const results = await session.run({ input_ids: tensorInput });
      const logitsTensor = results.logits; // shape [1, seq_len, vocab_size]

      const vocabSize = logitsTensor.dims[2];
      const seqLen = logitsTensor.dims[1];

      // Extract the logits for the last token in the sequence
      const lastTokenOffset = (seqLen - 1) * vocabSize;
      const lastTokenLogits = Array.from(
        logitsTensor.data.slice(lastTokenOffset, lastTokenOffset + vocabSize),
      );

      // Sample next token
      const nextTokenId = sampleFromLogits(lastTokenLogits, temp);
      inputIds.push(nextTokenId);

      // Decode and update UI live
      const currentText = tokenizer.decode(inputIds);
      modelMessageDiv.textContent = currentText;
      outputArea.scrollTop = outputArea.scrollHeight;
    }
  } catch (err) {
    console.error(err);
    modelMessageDiv.textContent += `\n[Error during generation: ${err.message}]`;
  } finally {
    isGenerating = false;
    generateBtn.disabled = false;
    promptInput.focus();
  }
}

// Event Listeners
generateBtn.addEventListener("click", generate);
promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") generate();
});

// Start initialization
window.addEventListener("DOMContentLoaded", initialize);

// Navigation UI Logic
const navItems = document.querySelectorAll('.nav-item');
const contentSections = document.querySelectorAll('.content-section');

navItems.forEach(item => {
  item.addEventListener('click', () => {
    // Remove active class from all nav items and sections
    navItems.forEach(nav => nav.classList.remove('active'));
    contentSections.forEach(section => section.classList.remove('active'));
    
    // Add active class to clicked item and corresponding section
    item.classList.add('active');
    const targetId = item.getAttribute('data-target');
    document.getElementById(targetId).classList.add('active');
  });
});

// --- Interactive Widgets Logic ---

// 1. Softmax Temperature & Top-K Widget
const widgetTemp = document.getElementById('widget-temp');
const widgetTempVal = document.getElementById('widget-temp-val');
const widgetTopK = document.getElementById('widget-topk');
const widgetTopKVal = document.getElementById('widget-topk-val');
const softmaxBars = document.getElementById('softmax-bars');

// Simulated raw logit outputs from a model
const rawLogits = [
  { word: "apple", logit: 2.5 },
  { word: "banana", logit: 1.2 },
  { word: "cat", logit: 0.1 },
  { word: "dog", logit: -1.5 }
];

function updateSoftmaxWidget() {
  const T = parseFloat(widgetTemp.value);
  const K = parseInt(widgetTopK.value);
  widgetTempVal.textContent = T.toFixed(1);
  widgetTopKVal.textContent = K === 4 ? "4 (All)" : K;
  
  // Apply Temperature
  let scaledLogits = rawLogits.map(item => item.logit / T);
  
  // Apply Top-K
  const sorted = [...scaledLogits].sort((a,b) => b - a);
  const threshold = sorted[K - 1];
  scaledLogits = scaledLogits.map(v => v >= threshold ? v : -Infinity);
  
  // Softmax
  const validLogits = scaledLogits.filter(v => v !== -Infinity);
  const maxLogit = validLogits.length > 0 ? Math.max(...validLogits) : 0;
  
  const exps = scaledLogits.map(v => v === -Infinity ? 0 : Math.exp(v - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(v => sumExps > 0 ? v / sumExps : 0);
  
  // Update UI
  softmaxBars.innerHTML = '';
  rawLogits.forEach((item, i) => {
    const probPercent = (probs[i] * 100).toFixed(1);
    
    const row = document.createElement('div');
    row.className = 'bar-row';
    
    row.innerHTML = `
      <div class="bar-label">${item.word}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${probPercent}%"></div>
      </div>
      <div class="bar-value">${probPercent}%</div>
    `;
    softmaxBars.appendChild(row);
  });
}

if(widgetTemp) {
  widgetTemp.addEventListener('input', updateSoftmaxWidget);
  widgetTopK.addEventListener('input', updateSoftmaxWidget);
  updateSoftmaxWidget(); // init
}

// 2. Causal Attention Mask Widget
const maskBtn = document.getElementById('toggle-mask-btn');
const attnGrid = document.getElementById('attn-grid');
let isMasked = false;

function initAttnGrid() {
  if(!attnGrid) return;
  attnGrid.innerHTML = '';
  for(let r=0; r<4; r++) {
    for(let c=0; c<4; c++) {
      const cell = document.createElement('div');
      cell.className = 'attn-cell';
      cell.dataset.r = r;
      cell.dataset.c = c;
      // Random attention score between 0.10 and 0.60
      cell.textContent = (Math.random() * 0.5 + 0.1).toFixed(2);
      attnGrid.appendChild(cell);
    }
  }
}

function toggleMask() {
  isMasked = !isMasked;
  maskBtn.textContent = isMasked ? "Remove Causal Mask" : "Apply Causal Mask (-∞)";
  const cells = attnGrid.querySelectorAll('.attn-cell');
  
  cells.forEach(cell => {
    const r = parseInt(cell.dataset.r);
    const c = parseInt(cell.dataset.c);
    
    // Upper triangle (future tokens)
    if(c > r) {
      if(isMasked) {
        cell.classList.add('masked');
        cell.textContent = '-∞';
      } else {
        cell.classList.remove('masked');
        cell.textContent = (Math.random() * 0.5 + 0.1).toFixed(2);
      }
    }
  });
}

if(maskBtn) {
  initAttnGrid();
  maskBtn.addEventListener('click', toggleMask);
}

// 3. Byte-Pair Encoding (BPE) Widget
const bpeTokensContainer = document.getElementById('bpe-tokens');
const bpeStepBtn = document.getElementById('bpe-step-btn');
const bpeResetBtn = document.getElementById('bpe-reset-btn');

// Initial string broken down into bytes/chars
const initialText = "h e l l o _ t h e r e".split(' ');
let currentTokens = [...initialText];

// Simulated prioritized merge rules learned from a corpus
const mergeRules = [
  ["e", "r", "er"],
  ["h", "e", "he"],
  ["l", "l", "ll"],
  ["he", "ll", "hell"],
  ["hell", "o", "hello"],
  ["t", "he", "the"],
  ["the", "er", "there"],
  ["_", "there", " there"], // Assuming _ is whitespace char
  ["hello", " there", "hello there"]
];
let currentStepIndex = 0;

function renderBpeTokens() {
  if(!bpeTokensContainer) return;
  bpeTokensContainer.innerHTML = '';
  currentTokens.forEach(t => {
    const div = document.createElement('div');
    div.className = 'bpe-token';
    div.textContent = t === "_" ? " " : t; // render _ as space for clarity if needed, or keep _
    bpeTokensContainer.appendChild(div);
  });
}

function bpeStep() {
  if (currentStepIndex >= mergeRules.length) return;
  
  const rule = mergeRules[currentStepIndex];
  let newTokens = [];
  let i = 0;
  
  // Apply rule across sequence
  while(i < currentTokens.length) {
    if (i < currentTokens.length - 1 && currentTokens[i] === rule[0] && currentTokens[i+1] === rule[1]) {
      newTokens.push(rule[2]);
      i += 2;
    } else {
      newTokens.push(currentTokens[i]);
      i += 1;
    }
  }
  
  currentTokens = newTokens;
  currentStepIndex++;
  renderBpeTokens();
  
  if (currentStepIndex >= mergeRules.length) {
    bpeStepBtn.disabled = true;
    bpeStepBtn.textContent = "Fully Merged!";
    bpeStepBtn.style.opacity = "0.5";
  }
}

function bpeReset() {
  currentTokens = [...initialText];
  currentStepIndex = 0;
  bpeStepBtn.disabled = false;
  bpeStepBtn.textContent = "Next Merge Step";
  bpeStepBtn.style.opacity = "1";
  renderBpeTokens();
}

if(bpeStepBtn) {
  renderBpeTokens();
  bpeStepBtn.addEventListener('click', bpeStep);
  bpeResetBtn.addEventListener('click', bpeReset);
}
