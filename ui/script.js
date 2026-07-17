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
