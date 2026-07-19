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

// 4. Attention Visualization Widget
const attnQueryWords = document.querySelectorAll('.attn-word.q-word');
const attnKeyWords = document.getElementById('attn-key-row');
const attnLinesContainer = document.getElementById('attn-lines');

function drawAttentionLines(activeIdx) {
  if(!attnLinesContainer || !attnKeyWords) return;
  attnLinesContainer.innerHTML = '';
  
  const containerRect = attnLinesContainer.getBoundingClientRect();
  // CRITICAL: Set SVG viewBox so the coordinate system matches the physical pixel dimensions
  attnLinesContainer.setAttribute('viewBox', `0 0 ${containerRect.width} ${containerRect.height}`);
  
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
  for(let i = 0; i < kChildren.length; i++) {
    const kWord = kChildren[i];
    
    // Masking future tokens
    if (i > activeIdx) {
      kWord.style.opacity = '0.3';
      kWord.style.filter = 'grayscale(100%)';
      continue; // Do not draw line
    } else {
      kWord.style.opacity = '1';
      kWord.style.filter = 'none';
    }
    
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
    
    attnLinesContainer.appendChild(line);
  }
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
  const colors = [
    'rgba(99, 102, 241, 0.3)',
    'rgba(139, 92, 246, 0.3)',
    'rgba(236, 72, 153, 0.3)',
    'rgba(14, 165, 233, 0.3)',
    'rgba(16, 185, 129, 0.3)'
  ];

  let mergeTimeout;
  let animationInterval;
  let innerTimeout;
  
  tokenizerInput.addEventListener('input', (e) => {
    clearTimeout(mergeTimeout);
    clearInterval(animationInterval);
    clearTimeout(innerTimeout);
    
    mergeTimeout = setTimeout(() => {
      if (!tokenizer) return;
      const text = e.target.value;
      if (!text) {
        bpeVisualizer.innerHTML = '<span style="color: var(--text-muted);">Start typing to see character merges...</span>';
        tokenizerOutput.innerHTML = '';
        tokenizerCount.textContent = '0';
        return;
      }
      
      const tokens = tokenizer.encode(text);
      let displayIds = [...tokens];
      if(displayIds.length > 0 && displayIds[0] === tokenizer.clsTokenId) displayIds.shift();
      
      tokenizerCount.textContent = displayIds.length;
      tokenizerOutput.innerHTML = '';
      
      // Prepare final chips for output area
      const finalChips = [];
      const targetSubwords = [];
      
      displayIds.forEach((id, index) => {
        const tokenStr = tokenizer.idToToken[id] || "[UNK]";
        let displayStr = tokenStr.startsWith("Ġ") ? " " + tokenStr.substring(1) : tokenStr;
        // Clean up text for BPE logic (no space replacement, just literal)
        let rawStr = tokenStr.replace('Ġ', ' ');
        
        targetSubwords.push(rawStr);
        
        const chip = document.createElement('div');
        chip.style.backgroundColor = colors[index % colors.length];
        chip.style.border = `1px solid ${colors[index % colors.length].replace('0.3', '0.6')}`;
        chip.style.padding = '0.4rem 0.8rem';
        chip.style.borderRadius = '6px';
        chip.style.fontFamily = 'monospace';
        chip.style.display = 'flex';
        chip.style.flexDirection = 'column';
        chip.style.alignItems = 'center';
        
        chip.innerHTML = `
          <span style="font-size: 1.1rem; color: white;">${displayStr.replace(/ /g, "␣").replace(/</g, '&lt;')}</span>
          <span style="font-size: 0.75rem; color: rgba(255,255,255,0.6); margin-top: 0.2rem;">ID: ${id}</span>
        `;
        finalChips.push(chip);
      });
      
      // Animate merges in bpeVisualizer
      // 1. Break everything into characters tagged with their target token index
      let currentSequence = [];
      targetSubwords.forEach((targetStr, targetIdx) => {
        let chars = targetStr.split('');
        chars.forEach(c => {
          currentSequence.push({ str: c, targetIdx: targetIdx });
        });
      });
      
      function renderSeq(seq, activeMergeIndex = -1) {
        bpeVisualizer.innerHTML = seq.map((s, i) => {
          const isActive = (i === activeMergeIndex || i === activeMergeIndex + 1);
          const bg = isActive ? 'rgba(99, 102, 241, 0.6)' : 'transparent';
          const border = isActive ? 'var(--primary)' : 'var(--glass-border)';
          return `<span style="border:1px solid ${border}; background: ${bg}; padding:0.2rem 0.4rem; border-radius:4px; transition: all 0.15s;">${s.str.replace(/ /g, '␣')}</span>`;
        }).join(' ');
      }
      
      renderSeq(currentSequence);
      
      animationInterval = setInterval(() => {
        // Find all possible merges
        let possibleMerges = [];
        for (let i = 0; i < currentSequence.length - 1; i++) {
          if (currentSequence[i].targetIdx === currentSequence[i+1].targetIdx) {
            possibleMerges.push(i);
          }
        }
        
        if (possibleMerges.length > 0) {
          // Pick a random valid merge to simulate frequency-based BPE (not just left-to-right)
          let mergeIdx = possibleMerges[Math.floor(Math.random() * possibleMerges.length)];
          
          // Highlight step
          renderSeq(currentSequence, mergeIdx);
          
          innerTimeout = setTimeout(() => {
            // Perform merge
            let mergedStr = currentSequence[mergeIdx].str + currentSequence[mergeIdx+1].str;
            let targetIdx = currentSequence[mergeIdx].targetIdx;
            currentSequence.splice(mergeIdx, 2, { str: mergedStr, targetIdx: targetIdx });
            renderSeq(currentSequence);
            
            if (currentSequence.length === targetSubwords.length) {
              clearInterval(animationInterval);
              innerTimeout = setTimeout(() => {
                finalChips.forEach(c => tokenizerOutput.appendChild(c));
              }, 150);
            }
          }, 150);
          
        } else {
          // Finished early or no merges needed
          clearInterval(animationInterval);
          finalChips.forEach(c => tokenizerOutput.appendChild(c));
        }
      }, 350);
    }, 500);
  });
}

// 6. Step-by-Step BPE Widget
const bpeTokensContainer = document.getElementById('bpe-tokens');
const bpeStepBtn = document.getElementById('bpe-step-btn');
const bpeResetBtn = document.getElementById('bpe-reset-btn');

const initialText = "h e l l o _ t h e r e".split(' ');
let currentTokens = [...initialText];

const mergeRules = [
  ["e", "r", "er"],
  ["h", "e", "he"],
  ["l", "l", "ll"],
  ["he", "ll", "hell"],
  ["hell", "o", "hello"],
  ["t", "he", "the"],
  ["the", "er", "there"],
  ["_", "there", " there"], 
  ["hello", " there", "hello there"]
];
let currentStepIndex = 0;

function renderBpeTokens() {
  if(!bpeTokensContainer) return;
  bpeTokensContainer.innerHTML = '';
  currentTokens.forEach(t => {
    const div = document.createElement('div');
    div.style.border = '1px solid var(--primary)';
    div.style.padding = '0.3rem 0.6rem';
    div.style.borderRadius = '4px';
    div.style.backgroundColor = 'rgba(99, 102, 241, 0.1)';
    div.textContent = t === "_" ? " " : t;
    bpeTokensContainer.appendChild(div);
  });
}

function bpeStep() {
  if (currentStepIndex >= mergeRules.length) return;
  
  const rule = mergeRules[currentStepIndex];
  let newTokens = [];
  let i = 0;
  
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

// RoPE Visualizer
const ropeSlider = document.getElementById('rope-slider');
const ropePosVal = document.getElementById('rope-pos-val');
const ropeCanvas = document.getElementById('rope-canvas');
if (ropeSlider && ropeCanvas) {
  const ctx = ropeCanvas.getContext('2d');
  
  function drawRoPE(pos) {
    ctx.clearRect(0, 0, 200, 200);
    const cx = 100, cy = 100;
    
    // Draw grid/axes
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, cy); ctx.lineTo(200, cy);
    ctx.moveTo(cx, 0); ctx.lineTo(cx, 200);
    ctx.stroke();
    
    // Draw bounds circle
    const r = 80;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI*2);
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]); // Reset
    
    // Base vector (Position 0, m=0)
    // Let's say it starts at angle Pi/6
    const baseAngle = Math.PI / 6; 
    const base_x = cx + r * Math.cos(baseAngle);
    const base_y = cy - r * Math.sin(baseAngle);
    
    // Draw original point vector (faint)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(base_x, base_y);
    ctx.stroke();
    
    // Draw original point
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.beginPath();
    ctx.arc(base_x, base_y, 4, 0, Math.PI*2);
    ctx.fill();
    
    // Rotation logic: Each position rotates it by some theta
    const theta = Math.PI / 8;
    const currentAngle = baseAngle + (pos * theta);
    
    const x = cx + r * Math.cos(currentAngle);
    const y = cy - r * Math.sin(currentAngle);
    
    // Draw rotation arc if pos > 0
    if (pos > 0) {
      ctx.beginPath();
      // Canvas arc needs negative angles for standard math coordinates
      ctx.arc(cx, cy, r/2, -currentAngle, -baseAngle, false);
      ctx.strokeStyle = 'rgba(236, 72, 153, 0.6)'; // Pink arc
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    
    // Draw rotated vector
    ctx.strokeStyle = '#6366f1'; // Primary color
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    // Draw rotated point
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
const rmsBarsIn = document.getElementById('rmsnorm-bars-in');
const rmsBarsOut = document.getElementById('rmsnorm-bars-out');

if (rmsSlider && rmsBarsIn && rmsBarsOut) {
  function drawRMSBars(variance) {
    rmsBarsIn.innerHTML = '';
    rmsBarsOut.innerHTML = '';
    const baseVals = [0.2, -0.5, 0.8, -0.3, 0.6, -0.9, 0.4];
    
    // Calculate RMS
    const scaledVals = baseVals.map(v => v * variance);
    const rms = Math.sqrt(scaledVals.reduce((acc, val) => acc + val*val, 0) / scaledVals.length) || 1;
    const normVals = scaledVals.map(v => v / rms);
    
    function drawRow(container, vals, scaleFactor) {
      vals.forEach(v => {
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.width = '100%';
        row.style.height = '14px';
        row.style.marginBottom = '4px';

        const leftSpace = document.createElement('div');
        leftSpace.style.flex = '1';
        leftSpace.style.display = 'flex';
        leftSpace.style.justifyContent = 'flex-end';

        const rightSpace = document.createElement('div');
        rightSpace.style.flex = '1';
        rightSpace.style.display = 'flex';
        rightSpace.style.justifyContent = 'flex-start';

        const bar = document.createElement('div');
        const w = Math.min(100, Math.abs(v) * scaleFactor);
        bar.style.width = `${w}px`;
        bar.style.height = '100%';
        bar.style.background = v > 0 ? 'var(--primary)' : '#f43f5e';
        bar.style.borderRadius = v > 0 ? '0 4px 4px 0' : '4px 0 0 4px';

        if (v < 0) {
          leftSpace.appendChild(bar);
        } else {
          rightSpace.appendChild(bar);
        }

        row.appendChild(leftSpace);
        
        const centerLine = document.createElement('div');
        centerLine.style.width = '2px';
        centerLine.style.height = '18px';
        centerLine.style.background = 'rgba(255,255,255,0.2)';
        row.appendChild(centerLine);
        
        row.appendChild(rightSpace);
        container.appendChild(row);
      });
    }

    // Input varies widely (scale directly by variance * base_scale)
    drawRow(rmsBarsIn, scaledVals, 40);
    // Output remains stable (scale by base_scale)
    drawRow(rmsBarsOut, normVals, 40);
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

// SwiGLU Visualizer
const swigluGateSlider = document.getElementById('swiglu-gate-slider');
const swigluLinSlider = document.getElementById('swiglu-lin-slider');
const swigluGateVal = document.getElementById('swiglu-gate-val');
const swigluLinVal = document.getElementById('swiglu-lin-val');

const swigluSiluNode = document.getElementById('swiglu-silu-node');
const swigluLinNode = document.getElementById('swiglu-lin-node');
const swigluOutNode = document.getElementById('swiglu-out-node');

function updateSwiGLU() {
  if (!swigluGateSlider) return;
  const gateIn = parseFloat(swigluGateSlider.value);
  const linIn = parseFloat(swigluLinSlider.value);
  
  swigluGateVal.textContent = gateIn.toFixed(1);
  swigluLinVal.textContent = linIn.toFixed(1);
  
  // SiLU = x * sigmoid(x)
  const sigmoid = 1 / (1 + Math.exp(-gateIn));
  const siluOut = gateIn * sigmoid;
  
  const finalOut = siluOut * linIn;
  
  swigluSiluNode.textContent = siluOut.toFixed(2);
  swigluLinNode.textContent = linIn.toFixed(1);
  swigluOutNode.textContent = finalOut.toFixed(2);
  
  // Visual feedback: opacity based on gate activation
  const gateOpacity = Math.min(1, Math.max(0.1, (siluOut + 0.5) / 2)); 
  swigluSiluNode.style.opacity = gateOpacity;
  
  // Output brightness based on absolute magnitude
  const outOpacity = Math.min(1, Math.max(0.1, Math.abs(finalOut)/15));
  swigluOutNode.style.opacity = outOpacity;
}

if(swigluGateSlider) {
  swigluGateSlider.addEventListener('input', updateSwiGLU);
  swigluLinSlider.addEventListener('input', updateSwiGLU);
  updateSwiGLU();
}
