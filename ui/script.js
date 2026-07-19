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

      // Handle HuggingFace style byte-level BPE spaces 'Ġ'
      if (token.startsWith("Ġ")) {
        text += " " + token.substring(1);
      } else {
        // Add spaces between all tokens for visibility in this educational UI
        text += (text.length > 0 ? " " : "") + token;
      }
    }
    return text.trim();
  }
}

// UI Elements
const textDisplay = document.getElementById("text-display");
const predictionsArea = document.getElementById("predictions-area");
const tokenCandidates = document.getElementById("token-candidates");
const promptInput = document.getElementById("prompt-input");
const generateBtn = document.getElementById("generate-btn");
const resetBtn = document.getElementById("reset-btn");
const tempSlider = document.getElementById("temp-slider");
const tempVal = document.getElementById("temp-val");
const topkSlider = document.getElementById("topk-slider");
const topkVal = document.getElementById("topk-val");

// State
let session = null;
let tokenizer = null;
let isGenerating = false;
let currentInputIds = [];

// Update labels
if (tempSlider) {
  tempSlider.addEventListener(
    "input",
    (e) => (tempVal.textContent = parseFloat(e.target.value).toFixed(1)),
  );
}
if (topkSlider) {
  topkSlider.addEventListener(
    "input",
    (e) => (topkVal.textContent = e.target.value),
  );
}

// Example Chips
document.querySelectorAll('.example-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    if(!promptInput.disabled) {
      promptInput.value = chip.textContent;
      generateBtn.click();
    }
  });
});

async function initialize() {
  const initLoader = document.getElementById('init-loader');
  const initText = document.getElementById('init-text');
  const examplePrompts = document.getElementById('example-prompts');
  
  try {
    const tResp = await fetch("../tokenizer.json");
    const tJson = await tResp.json();
    tokenizer = new BasicTokenizer(tJson);

    if(initText) initText.innerHTML = 'Loading ONNX Runtime and Model parameters...';

    session = await ort.InferenceSession.create("../tiny_llm_quantized.onnx", {
      executionProviders: ["wasm"],
    });

    if(initLoader) initLoader.style.display = 'none';
    if(initText) {
      initText.innerHTML = '✅ Model loaded! Enter a prompt below or pick an example to start.';
      initText.style.color = 'var(--primary)';
    }
    if(examplePrompts) examplePrompts.style.display = 'block';
    
    promptInput.disabled = false;
    generateBtn.disabled = false;
    promptInput.focus();
    
    if(typeof updateTokenizerPlayground === 'function') {
      updateTokenizerPlayground();
    }
  } catch (err) {
    console.error(err);
    if(initLoader) initLoader.style.display = 'none';
    if(initText) initText.innerHTML = `<span style="color: red;">Error loading model: ${err.message}</span>`;
  }
}

async function startInteractive() {
  if (!promptInput.value.trim() || isGenerating || !session) return;

  const prompt = promptInput.value.trim();
  promptInput.disabled = true;
  generateBtn.disabled = true;
  if(resetBtn) resetBtn.disabled = false;
  
  currentInputIds = tokenizer.encode(prompt);
  textDisplay.textContent = prompt;
  predictionsArea.style.display = 'block';
  
  await predictNextTokens();
}

async function predictNextTokens() {
  if (!session) return;
  isGenerating = true;
  
  const temp = tempSlider ? parseFloat(tempSlider.value) : 0.8;
  tokenCandidates.innerHTML = '<div style="color: var(--text-muted);">Computing next tokens...</div>';

  try {
    const tensorInput = new ort.Tensor(
      "int64",
      BigInt64Array.from(currentInputIds.map(BigInt)),
      [1, currentInputIds.length],
    );

    const results = await session.run({ input_ids: tensorInput });
    const logitsTensor = results.logits; 
    const vocabSize = logitsTensor.dims[2];
    const seqLen = logitsTensor.dims[1];

    const lastTokenOffset = (seqLen - 1) * vocabSize;
    const lastTokenLogits = Array.from(
      logitsTensor.data.slice(lastTokenOffset, lastTokenOffset + vocabSize),
    );

    const scaledLogits = lastTokenLogits.map((v) => v / temp);
    const maxLogit = Math.max(...scaledLogits);
    const exps = scaledLogits.map((v) => Math.exp(v - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((v) => v / sumExps);

    const indexedProbs = probs.map((p, idx) => ({ prob: p, id: idx }));
    indexedProbs.sort((a, b) => b.prob - a.prob);
    
    const topk = topkSlider ? parseInt(topkSlider.value, 10) : 5;
    const topKItems = indexedProbs.slice(0, topk);

    tokenCandidates.innerHTML = '';
    topKItems.forEach(item => {
      const tokenStr = tokenizer.idToToken[item.id] || "[UNK]";
      const displayStr = tokenStr.startsWith(" ") ? " " + tokenStr.substring(1) : tokenStr.replace(/Ġ/g, " ");
      const probPercent = (item.prob * 100).toFixed(1);
      
      const btn = document.createElement('button');
      btn.className = 'token-candidate-btn';
      btn.innerHTML = `
        <div class="token-candidate-bg" style="width: ${probPercent}%"></div>
        <div class="token-candidate-text">${displayStr.replace(/</g, '&lt;')}</div>
        <div class="token-candidate-prob">${probPercent}%</div>
      `;
      
      btn.onclick = async () => {
        currentInputIds.push(item.id);
        textDisplay.textContent = tokenizer.decode(currentInputIds);
        await predictNextTokens();
      };
      
      tokenCandidates.appendChild(btn);
    });

  } catch (err) {
    console.error(err);
    tokenCandidates.innerHTML = `<div style="color: red;">Error: ${err.message}</div>`;
  } finally {
    isGenerating = false;
  }
}

// Event Listeners
if(generateBtn) generateBtn.addEventListener("click", startInteractive);
if(promptInput) {
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") startInteractive();
  });
}
if(resetBtn) {
  resetBtn.addEventListener('click', () => {
    promptInput.disabled = false;
    generateBtn.disabled = false;
    resetBtn.disabled = true;
    promptInput.value = "";
    textDisplay.innerHTML = '<span class="placeholder-text" style="color: var(--text-muted);">✅ Model loaded! Enter a prompt below to start.</span>';
    predictionsArea.style.display = 'none';
    currentInputIds = [];
  });
}

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

// 4. Attention Visualization Widget
const attnQueryWords = document.querySelectorAll('.attn-word.q-word');
const attnKeyWords = document.getElementById('attn-key-row');
const attnLinesContainer = document.getElementById('attn-lines');

function drawAttentionLines(activeIdx) {
  if(!attnLinesContainer || !attnKeyWords) return;
  attnLinesContainer.innerHTML = '';
  
  // Create SVG element
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.style.width = '100%';
  svg.style.height = '100%';
  
  const containerRect = attnLinesContainer.getBoundingClientRect();
  const qWord = attnQueryWords[activeIdx];
  const qRect = qWord.getBoundingClientRect();
  
  const startX = qRect.left - containerRect.left + (qRect.width / 2);
  const startY = qRect.top - containerRect.top; // top of query word
  
  // Randomize attention weights that sum to 1
  let weights = [];
  for(let i=0; i<=activeIdx; i++) weights.push(Math.random());
  const sumWeights = weights.reduce((a,b)=>a+b, 0);
  weights = weights.map(w => w / sumWeights);

  const kChildren = attnKeyWords.children;
  for(let i=0; i<=activeIdx; i++) {
    const kWord = kChildren[i];
    const kRect = kWord.getBoundingClientRect();
    const endX = kRect.left - containerRect.left + (kRect.width / 2);
    const endY = kRect.bottom - containerRect.top + 5; // bottom of key word
    
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', startX);
    line.setAttribute('y1', startY);
    line.setAttribute('x2', endX);
    line.setAttribute('y2', endY);
    
    // Line thickness based on attention weight
    const strokeWidth = Math.max(1, weights[i] * 12);
    const opacity = Math.max(0.2, weights[i] + 0.2);
    
    line.setAttribute('stroke', '#8b5cf6');
    line.setAttribute('stroke-width', strokeWidth);
    line.setAttribute('opacity', opacity);
    line.setAttribute('stroke-linecap', 'round');
    
    svg.appendChild(line);
  }
  
  attnLinesContainer.appendChild(svg);
}

if(attnQueryWords.length > 0) {
  attnQueryWords.forEach(word => {
    word.addEventListener('click', (e) => {
      attnQueryWords.forEach(w => w.classList.remove('active'));
      e.target.classList.add('active');
      const idx = parseInt(e.target.getAttribute('data-idx'));
      drawAttentionLines(idx);
    });
  });
  
  // Initial draw
  setTimeout(() => drawAttentionLines(3), 100);
  window.addEventListener('resize', () => {
    const active = document.querySelector('.attn-word.q-word.active');
    if(active) {
      drawAttentionLines(parseInt(active.getAttribute('data-idx')));
    }
  });
}



// 5. Tokenizer Playground Merges
const tokenizerInput = document.getElementById('tokenizer-playground-input');
const bpeVisualizer = document.getElementById('bpe-merge-visualizer');
const tokenizerOutput = document.getElementById('tokenizer-playground-output');
const tokenizerCount = document.getElementById('tokenizer-count');

if (tokenizerInput && bpeVisualizer && tokenizerOutput) {
  let mergeTimeout;
  tokenizerInput.addEventListener('input', (e) => {
    clearTimeout(mergeTimeout);
    mergeTimeout = setTimeout(() => {
      if (!tokenizer) return;
      const text = e.target.value;
      if (!text) {
        bpeVisualizer.innerHTML = '<span style="color: var(--text-muted);">Start typing to see character merges...</span>';
        tokenizerOutput.innerHTML = '';
        tokenizerCount.textContent = '0';
        return;
      }
      
      // Simulate merging
      const chars = Array.from(text).map(c => `<span style="border:1px solid var(--glass-border); padding:0.2rem 0.4rem; border-radius:4px;">${c}</span>`);
      bpeVisualizer.innerHTML = chars.join(' ');
      
      const tokens = tokenizer.encode(text);
      tokenizerCount.textContent = tokens.length;
      tokenizerOutput.innerHTML = tokens.map(t => `<span class="example-chip">${tokenizer.idToToken[t] || '[UNK]'}</span>`).join('');
      
    }, 500);
  });
}

// RoPE Visualizer
const ropeSlider = document.getElementById('rope-slider');
const ropePosVal = document.getElementById('rope-pos-val');
const ropeCanvas = document.getElementById('rope-canvas');
if (ropeSlider && ropeCanvas) {
  const ctx = ropeCanvas.getContext('2d');
  function drawRoPE(pos) {
    ctx.clearRect(0, 0, 200, 200);
    const cx = 100, cy = 100;
    
    // Draw axes
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.beginPath();
    ctx.moveTo(0, cy); ctx.lineTo(200, cy);
    ctx.moveTo(cx, 0); ctx.lineTo(cx, 200);
    ctx.stroke();
    
    // Rotate vector
    const angle = pos * (Math.PI / 8); // example angle
    const r = 80;
    const x = cx + r * Math.cos(angle);
    const y = cy - r * Math.sin(angle);
    
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    ctx.fillStyle = '#818cf8';
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI*2);
    ctx.fill();
  }
  
  ropeSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    ropePosVal.textContent = val;
    drawRoPE(val);
  });
  
  drawRoPE(0);
}

// RMSNorm Visualizer
const rmsSlider = document.getElementById('rmsnorm-slider');
const rmsVal = document.getElementById('rmsnorm-val');
const rmsBars = document.getElementById('rmsnorm-bars');

if (rmsSlider && rmsBars) {
  function drawRMSBars(variance) {
    rmsBars.innerHTML = '';
    const baseVals = [0.2, -0.5, 0.8, -0.3, 0.6, -0.9, 0.4];
    
    // Calculate RMS
    const scaledVals = baseVals.map(v => v * variance);
    const rms = Math.sqrt(scaledVals.reduce((acc, val) => acc + val*val, 0) / scaledVals.length) || 1;
    const normVals = scaledVals.map(v => v / rms);
    
    normVals.forEach(v => {
      const h = Math.abs(v) * 40; // max height ~100px
      const bar = document.createElement('div');
      bar.style.width = '20px';
      bar.style.height = `${h}px`;
      bar.style.background = v > 0 ? 'var(--primary)' : '#f43f5e';
      bar.style.borderRadius = '4px 4px 0 0';
      rmsBars.appendChild(bar);
    });
  }
  
  rmsSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    rmsVal.textContent = val.toFixed(1);
    drawRMSBars(val);
  });
  
  drawRMSBars(1.0);
}

// Training Stepper
const trainStepBtn = document.getElementById('train-step-btn');
const trainSteps = document.querySelectorAll('.train-step');
let currentTrainStep = 1;

if (trainStepBtn) {
  trainStepBtn.addEventListener('click', () => {
    trainSteps.forEach(s => s.classList.remove('active', 'done'));
    
    currentTrainStep++;
    if(currentTrainStep > 6) currentTrainStep = 1;
    
    trainSteps.forEach((s, idx) => {
      if (idx + 1 < currentTrainStep) {
        s.classList.add('done');
      } else if (idx + 1 === currentTrainStep) {
        s.classList.add('active');
      }
    });
  });
}
