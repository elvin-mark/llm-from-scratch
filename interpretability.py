import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
import streamlit as st
import pandas as pd
import numpy as np

# Try importing plotly for interactive premium plots; fallback to matplotlib if not installed
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib
    matplotlib.use("Agg")  # Prevents segmentation faults on macOS by using a non-GUI backend
    import matplotlib.pyplot as plt
    import seaborn as sns

# Ensure the local directory is in the path so we can import tinyllm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import TinyLLM, apply_rotary_emb
except ImportError:
    st.error("Could not import `TinyLLM` from `model.py`. Please make sure `model.py` is in the same directory.")
    st.stop()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

def get_causal_mask(seqlen, device):
    if seqlen > 1:
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask
    return None

# -------------------------------------------------------------
# 1. Custom Attention Wrapper to Capture Attention Maps & Ablation
# -------------------------------------------------------------
class HookedAttention(nn.Module):
    def __init__(self, original_attention, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = original_attention.n_heads
        self.head_dim = original_attention.head_dim
        self.wq = original_attention.wq
        self.wk = original_attention.wk
        self.wv = original_attention.wv
        self.wo = original_attention.wo
        self.last_attention_weights = None

    def forward(self, x, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        self.last_attention_weights = weights.detach().cpu()

        # Apply head ablation: zero out attention weights for specific heads
        weights_modified = weights.clone()
        ablation_mask = st.session_state.get('ablation_mask', {})
        for head_idx in range(self.n_heads):
            if ablation_mask.get((self.layer_idx, head_idx), False):
                # Zero attention weights for this head
                weights_modified[:, head_idx, :, :] = 0.0

        output = torch.matmul(weights_modified, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

# -------------------------------------------------------------
# 2. Custom FeedForward Wrapper to Capture Neuron Activations (fMRI)
# -------------------------------------------------------------
class HookedFeedForward(nn.Module):
    def __init__(self, original_ffn, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.w1 = original_ffn.w1
        self.w2 = original_ffn.w2
        self.w3 = original_ffn.w3
        self.last_activations = None

    def forward(self, x):
        acts = F.silu(self.w1(x)) * self.w3(x)  # [bsz, seqlen, hidden_dim]
        self.last_activations = acts.detach().cpu()
        return self.w2(acts)

# -------------------------------------------------------------
# 3. Model & Tokenizer Loader
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and tokenizer...")
def load_model_and_tokenizer(model_path, tokenizer_path, dim, n_layers, n_heads, ffn_dim, max_seq_len):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    model = TinyLLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Hook attention and Feed-Forward layers
    for idx, block in enumerate(model.layers):
        if not isinstance(block.attention, HookedAttention):
            block.attention = HookedAttention(block.attention, layer_idx=idx)
        if not isinstance(block.feed_forward, HookedFeedForward):
            block.feed_forward = HookedFeedForward(block.feed_forward, layer_idx=idx)

    model = model.to(DEVICE)
    return model, tokenizer

# -------------------------------------------------------------
# 4. Logit Lens & Attribution Computation Helper
# -------------------------------------------------------------
def run_logit_lens_and_attribution(model, tokenizer, tokens):
    bsz, seqlen = tokens.shape
    h = model.tok_embeddings(tokens)
    freqs_cis = model.freqs_cis[:seqlen].to(DEVICE)

    mask = get_causal_mask(seqlen, DEVICE)

    layer_names = ["Embedding Layer"]
    layer_states = []

    layer_states.append(h)

    curr_h = h
    for idx, layer in enumerate(model.layers):
        curr_h = layer(curr_h, freqs_cis, mask)
        layer_names.append(f"Transformer Layer {idx}")
        layer_states.append(curr_h)

    predictions = []
    for state in layer_states:
        logits = model.output(model.norm(state))[0].detach()
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=1, dim=-1)

        layer_preds = []
        for seq_i in range(seqlen):
            tid = top_ids[seq_i].item()
            prob = top_probs[seq_i].item()
            layer_preds.append((tokenizer.id_to_token(tid), prob))
        predictions.append(layer_preds)

    final_state = layer_states[-1]
    final_normed = model.norm(final_state)
    final_logits = model.output(final_normed)[0, -1]

    chosen_token_id = torch.argmax(final_logits).item()
    chosen_token_str = tokenizer.id_to_token(chosen_token_id)

    prev_logit = 0.0
    contribs = []
    for idx, state in enumerate(layer_states):
        with torch.no_grad():
            state_logits = model.output(model.norm(state))[0, -1]
            logit_val = state_logits[chosen_token_id].item()

        delta = logit_val - prev_logit if idx > 0 else logit_val
        prev_logit = logit_val

        contribs.append(
            {
                "Layer": layer_names[idx],
                "Delta Contribution (Logit Units)": round(delta, 3),
                "Cumulative Logit Value": round(logit_val, 3),
            }
        )

    return layer_names, predictions, chosen_token_str, contribs

# -------------------------------------------------------------
# 5. Gradient-based Saliency Attribution
# -------------------------------------------------------------
def run_gradient_saliency(model, tokenizer, tokens):
    model.zero_grad()
    embeddings = model.tok_embeddings.weight
    bsz, seqlen = tokens.shape

    embeddings_grad = embeddings.clone().detach().requires_grad_(True)
    h = embeddings_grad[tokens[0]].unsqueeze(0)

    freqs_cis = model.freqs_cis[:seqlen].to(DEVICE)
    mask = get_causal_mask(seqlen, DEVICE)

    curr_h = h
    for layer in model.layers:
        curr_h = layer(curr_h, freqs_cis, mask)
    curr_h = model.norm(curr_h)
    logits = model.output(curr_h)

    last_position_logits = logits[0, -1, :]
    top_class_id = torch.argmax(last_position_logits).item()
    target_score = last_position_logits[top_class_id]

    target_score.backward()

    grad_at_embeddings = embeddings_grad.grad
    if grad_at_embeddings is not None:
        position_saliency = torch.norm(grad_at_embeddings[tokens[0]], dim=-1).cpu().numpy()
        max_val = position_saliency.max() + 1e-8
        normalized_saliency = position_saliency / max_val
        return normalized_saliency
    else:
        return np.ones(seqlen)

# -------------------------------------------------------------
# Streamlit Dashboard UI Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="TinyLLM Interpretability Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    h1 { color: #4F46E5 !important; font-weight: 800; }
    h2, h3 { color: #818CF8 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background-color: #1F2937;
        border-radius: 4px;
        color: #9CA3AF;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F46E5 !important;
        color: white !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 TinyLLM Mechanistic Interpretability Dashboard")
st.write("Explore how your trained TinyLLM model represents and processes language.")

if 'ablation_mask' not in st.session_state:
    st.session_state.ablation_mask = {}

# Sidebar Configuration
st.sidebar.header("📁 Model Configurations")
model_file = st.sidebar.text_input("Model Weights Path", "tiny_llm.pth")
tokenizer_file = st.sidebar.text_input("Tokenizer Path", "tokenizer.json")

st.sidebar.subheader("Architecture")
dim = st.sidebar.number_input("Dimension (dim)", min_value=16, value=128)
n_layers = st.sidebar.number_input("Num Layers (n_layers)", min_value=1, value=4)
n_heads = st.sidebar.number_input("Num Heads (n_heads)", min_value=1, value=4)
ffn_dim = st.sidebar.number_input("FFN Dim (ffn_dim)", min_value=16, value=512)
max_seq_len = st.sidebar.number_input("Max Seq Length", min_value=16, value=64)

if not os.path.exists(model_file) or not os.path.exists(tokenizer_file):
    st.sidebar.error("⚠️ Model file or Tokenizer file not found. Please verify the absolute paths.")
    st.stop()
else:
    try:
        model, tokenizer = load_model_and_tokenizer(model_file, tokenizer_file, dim, n_layers, n_heads, ffn_dim, max_seq_len)
        st.sidebar.success(f"✅ Model and Tokenizer loaded successfully on {DEVICE.type.upper()}!")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        st.stop()

# Head Ablation Settings on Sidebar
st.sidebar.divider()
st.sidebar.subheader("✂️ Attention Head Ablation")
st.sidebar.write("Toggle specific heads to ablate (turn off) them inside the model dynamically during calculation.")

for l_i in range(n_layers):
    cols_heads = st.sidebar.columns(min(n_heads, 4))
    for h_i in range(n_heads):
        with cols_heads[h_i % 4]:
            key = f"L{l_i}H{h_i}"
            is_checked = st.checkbox(key, value=st.session_state.ablation_mask.get((l_i, h_i), False), help=f"Ablate Layer {l_i} Head {h_i}")
            st.session_state.ablation_mask[(l_i, h_i)] = is_checked

# Tab Layout
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🕸️ Self-Attention Mapping",
        "🔍 Logit Lens & Layer Attribution",
        "🧠 LLM fMRI (Neuron Activations)",
        "🎯 Causal Word Saliency (Gradients)",
        "📊 Predictability & Entropy",
        "🎮 Interactive Generator",
    ]
)

def validate_input_sequence(text):
    if not text.strip():
        st.warning("Please enter some text.")
        return None, None
    cls_id = tokenizer.token_to_id("[CLS]")
    encoded = tokenizer.encode(text)
    if not encoded:
        st.warning("Could not encode text.")
        return None, None
    tokens = [cls_id] + encoded.ids
    if len(tokens) > max_seq_len:
        st.warning(f"Input is too long ({len(tokens)} tokens). Truncating to {max_seq_len} tokens.")
        tokens = tokens[:max_seq_len]
    return tokens, [tokenizer.id_to_token(t) for t in tokens]

with tab1:
    st.header("Attention Maps Visualization")
    st.write("Visualize the self-attention weights inside the transformer layers to see which words route information to each other.")

    col1, col2 = st.columns([1, 3])
    with col1:
        test_sentence = st.text_input("Enter a sentence to analyze:", "오늘 날씨가 아주 좋습니다", key="tab1_input")
        layer_selection = st.selectbox("Select Layer:", range(n_layers), index=n_layers-1, key="tab1_layer")
        head_selection = st.selectbox("Select Attention Head:", range(n_heads), index=0, key="tab1_head")

        tokens, token_strs = validate_input_sequence(test_sentence)

    with col2:
        if tokens:
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                model(input_tensor)

            block = model.layers[layer_selection]
            attn_matrix = block.attention.last_attention_weights[0, head_selection].numpy()

            if HAS_PLOTLY:
                fig = go.Figure(data=go.Heatmap(z=attn_matrix, x=token_strs, y=token_strs, colorscale="Cividis"))
                fig.update_layout(
                    title=f"Attention Map (Layer {layer_selection}, Head {head_selection})",
                    xaxis_title="Key Tokens (Attended To)",
                    yaxis_title="Query Tokens (Attending)",
                    template="plotly_dark",
                    width=700, height=550,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(attn_matrix, xticklabels=token_strs, yticklabels=token_strs, cmap="cividis", annot=True, fmt=".2f", ax=ax)
                st.pyplot(fig)

with tab2:
    st.header("The Logit Lens & Residual Attribution")
    lens_sentence = st.text_input("Enter a sentence to pass through the lens:", "오늘 날씨가 아주 좋습니다", key="tab2_lens_input")
    lens_tokens, lens_token_strs = validate_input_sequence(lens_sentence)

    if lens_tokens:
        input_tensor = torch.tensor([lens_tokens], dtype=torch.long).to(DEVICE)
        layer_names, predictions, final_token, attribs = run_logit_lens_and_attribution(model, tokenizer, input_tensor)

        st.subheader(f"1. Predictions Grid (Final Token Predicted: `{final_token}`)")
        
        def render_logit_lens_html(layer_names, token_strs, predictions):
            html = "<div style='overflow-x: auto;'><table style='width:100%; border-collapse: collapse; color: white; background-color: #1F2937; min-width: 600px;'>"
            html += "<tr style='border-bottom: 2px solid #4F46E5;'>"
            html += "<th style='padding: 12px; text-align: left; background-color: #111827;'>Layer</th>"
            for token in token_strs:
                html += f"<th style='padding: 12px; text-align: center; background-color: #111827;'>{token}</th>"
            html += "</tr>"
            for l_idx, layer_name in enumerate(layer_names):
                html += "<tr style='border-bottom: 1px solid #374151;'>"
                html += f"<td style='padding: 12px; font-weight: bold; background-color: #111827; white-space: nowrap;'>{layer_name}</td>"
                for col_idx in range(len(token_strs)):
                    word, prob = predictions[l_idx][col_idx]
                    opacity = min(max(prob, 0.05), 1.0)
                    bg_color = f"rgba(79, 70, 229, {opacity})"
                    html += f"<td style='padding: 12px; text-align: center; background-color: {bg_color}; border: 1px solid #374151; min-width: 80px;'>"
                    html += f"<div style='font-weight: bold; font-size: 14px;'>{word}</div>"
                    html += f"<div style='font-size: 11px; opacity: 0.8;'>{prob * 100:.1f}%</div>"
                    html += "</td>"
                html += "</tr>"
            html += "</table></div>"
            return html

        st.markdown(render_logit_lens_html(layer_names, lens_token_strs, predictions), unsafe_allow_html=True)

        st.divider()
        st.subheader(f"2. Layer-by-Layer Attribution for Predicted Token: `{final_token}`")
        df_attribs = pd.DataFrame(attribs)
        col_chart, col_explain = st.columns([2, 1])

        with col_chart:
            if HAS_PLOTLY:
                fig_waterfall = go.Figure(go.Waterfall(
                    name="Attribution", orientation="v",
                    measure=["relative"] * len(df_attribs),
                    x=df_attribs["Layer"],
                    textposition="outside",
                    text=df_attribs["Delta Contribution (Logit Units)"].astype(str),
                    y=df_attribs["Delta Contribution (Logit Units)"],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                ))
                fig_waterfall.update_layout(title=f"Logit Contribution Breakdown (Why did model choose '{final_token}'?)", template="plotly_dark", height=400)
                st.plotly_chart(fig_waterfall, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(data=df_attribs, x="Layer", y="Delta Contribution (Logit Units)", palette="coolwarm", ax=ax)
                st.pyplot(fig)

        with col_explain:
            st.dataframe(df_attribs, use_container_width=True)

with tab3:
    st.header("🧠 LLM fMRI: Neuron Activation Firing")
    fmri_sentence = st.text_input("Enter a sentence for fMRI scan:", "오늘 날씨가 아주 좋습니다", key="tab3_fmri_input")
    fmri_tokens, fmri_token_strs = validate_input_sequence(fmri_sentence)

    if fmri_tokens:
        input_tensor = torch.tensor([fmri_tokens], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            model(input_tensor)

        fmri_layer_idx = st.selectbox("Select Layer for fMRI scan:", range(n_layers), index=min(2, n_layers-1), key="tab3_fmri_layer")
        block = model.layers[fmri_layer_idx]
        activations = block.feed_forward.last_activations[0].numpy()  # [seqlen, ffn_dim]

        num_neurons_to_show = st.slider("Number of active neurons to display:", 10, min(100, ffn_dim), min(30, ffn_dim))
        peak_activations = activations.max(axis=0)
        top_neuron_indices = np.argsort(peak_activations)[::-1][:num_neurons_to_show]
        filtered_activations = activations[:, top_neuron_indices].T
        neuron_labels = [f"Neuron {idx}" for idx in top_neuron_indices]

        if HAS_PLOTLY:
            fig_fmri = go.Figure(data=go.Heatmap(z=filtered_activations, x=fmri_token_strs, y=neuron_labels, colorscale="Hot"))
            fig_fmri.update_layout(title=f"fMRI Brain Scan of FFN layer {fmri_layer_idx} (Top {num_neurons_to_show} Neurons)", template="plotly_dark", height=200 + num_neurons_to_show * 15)
            st.plotly_chart(fig_fmri, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(filtered_activations, xticklabels=fmri_token_strs, yticklabels=neuron_labels, cmap="hot", ax=ax)
            st.pyplot(fig)

        col_tok, col_neu = st.columns(2)
        with col_tok:
            selected_token = st.selectbox("Select Word:", fmri_token_strs, index=min(1, len(fmri_token_strs) - 1))
            token_pos = fmri_token_strs.index(selected_token)
            token_acts = activations[token_pos]
            top_token_neurons = np.argsort(token_acts)[::-1][:10]
            df_tok_neu = pd.DataFrame({"Neuron": [f"Neuron #{idx}" for idx in top_token_neurons], "Activation Value": np.round(token_acts[top_token_neurons], 3)})
            
            if HAS_PLOTLY:
                fig_tok_neu = px.bar(df_tok_neu, x="Activation Value", y="Neuron", orientation="h", title=f"Top 10 Neurons Firing for '{selected_token}'", color="Activation Value", color_continuous_scale="Reds")
                fig_tok_neu.update_layout(yaxis={"categoryorder": "total ascending"}, template="plotly_dark", height=300)
                st.plotly_chart(fig_tok_neu, use_container_width=True)
            else:
                st.dataframe(df_tok_neu)

        with col_neu:
            neuron_id = st.number_input(f"Enter Neuron ID (0 to {ffn_dim-1}):", min_value=0, max_value=ffn_dim-1, value=int(top_neuron_indices[0]))
            df_neu_acts = pd.DataFrame({"Token": fmri_token_strs, "Activation Firing": np.round(activations[:, neuron_id], 3)})
            
            if HAS_PLOTLY:
                fig_neu_acts = px.bar(df_neu_acts, x="Token", y="Activation Firing", title=f"Neuron #{neuron_id} Firing Pattern Across Sentence", color="Activation Firing", color_continuous_scale="Oranges")
                fig_neu_acts.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig_neu_acts, use_container_width=True)
            else:
                st.dataframe(df_neu_acts)

with tab4:
    st.header("Causal Word Saliency Mapping")
    saliency_sentence = st.text_input("Enter a sentence to analyze:", "오늘 날씨가 아주 좋습니다", key="tab4_input")
    saliency_tokens, saliency_token_strs = validate_input_sequence(saliency_sentence)

    if saliency_tokens:
        input_tensor = torch.tensor([saliency_tokens], dtype=torch.long).to(DEVICE)
        normalized_saliency = run_gradient_saliency(model, tokenizer, input_tensor)

        with torch.no_grad():
            final_preds = model(input_tensor)
        target_token_id = torch.argmax(final_preds[0, -1, :]).item()
        target_token_str = tokenizer.id_to_token(target_token_id)

        st.markdown(f"### Causal importance for predicting: **`{target_token_str}`**")
        html_saliency = "<div style='display: flex; gap: 8px; flex-wrap: wrap; padding: 20px; background-color: #111827; border-radius: 8px; border: 1px solid #374151;'>"
        for word_i, token_str in enumerate(saliency_token_strs):
            score = normalized_saliency[word_i]
            bg_color = f"rgba(79, 70, 229, {score * 0.9 + 0.1})"
            html_saliency += f"<div style='padding: 8px 16px; background-color: {bg_color}; border-radius: 4px; border: 1px solid rgba(255,255,255,0.1); text-align: center; color: white;'><div style='font-weight: bold; font-size: 16px;'>{token_str}</div><div style='font-size: 11px; opacity: 0.8;'>{score:.2f}</div></div>"
        html_saliency += "</div>"
        st.markdown(html_saliency, unsafe_allow_html=True)

with tab5:
    st.header("Next-Token Predictability & Entropy Profile")
    if 'tokens' in locals() and tokens: # Use tokens from tab1 or re-evaluate
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = model(input_tensor)

        probs = F.softmax(logits[0], dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1).cpu().numpy()

        analysis_data = []
        for idx, token in enumerate(token_strs[:-1]):
            next_token = token_strs[idx + 1]
            next_token_id = tokens[idx + 1]
            confidence = probs[idx, next_token_id].item()
            analysis_data.append({"Position": idx, "Context Word": token, "Predicted Word": next_token, "Surprisal (Entropy in bits)": round(entropy[idx], 2), "Confidence Probability (%)": round(confidence * 100, 2)})

        df_analysis = pd.DataFrame(analysis_data)
        col1, col2 = st.columns([2, 1])

        with col1:
            if HAS_PLOTLY:
                fig = px.line(df_analysis, x="Context Word", y="Surprisal (Entropy in bits)", markers=True, title="Uncertainty (Entropy) Profile across the Sentence")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(data=df_analysis, x="Context Word", y="Surprisal (Entropy in bits)", marker="o", ax=ax)
                st.pyplot(fig)
        with col2:
            st.dataframe(df_analysis, use_container_width=True)

with tab6:
    st.header("Interactive Generation & Decoding Steering")
    cls_id = tokenizer.token_to_id("[CLS]") if 'tokenizer' in locals() else 0
    if "gen_token_ids" not in st.session_state:
        st.session_state.gen_token_ids = [cls_id]

    col1, col2 = st.columns([1, 2])
    with col1:
        temperature = st.slider("Temperature:", 0.1, 2.0, 0.8, step=0.1)
        top_k_val = st.slider("Top-K:", 1, 100, 50, step=1)
        custom_prompt = st.text_input("Reset with custom prompt:", "")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Apply Prompt & Reset"):
                if custom_prompt.strip():
                    encoded = tokenizer.encode(custom_prompt)
                    ids = encoded.ids[:max_seq_len-1] # ensure it doesn't exceed limit
                    st.session_state.gen_token_ids = [cls_id] + ids
                else:
                    st.session_state.gen_token_ids = [cls_id]
                st.rerun()
        with col_btn2:
            if st.button("Reset to [CLS]"):
                st.session_state.gen_token_ids = [cls_id]
                st.rerun()

        if st.button("🤖 Auto-Generate 1 Token"):
            if len(st.session_state.gen_token_ids) >= max_seq_len:
                st.warning("Maximum sequence length reached.")
            else:
                current_ids = torch.tensor([st.session_state.gen_token_ids], dtype=torch.long).to(DEVICE)
                with torch.no_grad():
                    logits = model(current_ids)
                    next_token_logits = logits[0, -1, :] / temperature
                    if top_k_val > 0:
                        top_k_threshold = torch.topk(next_token_logits, top_k_val)[0][-1]
                        next_token_logits[next_token_logits < top_k_threshold] = float("-inf")
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    st.session_state.gen_token_ids.append(next_token)
                    st.rerun()

    with col2:
        token_strs_gen = [tokenizer.id_to_token(tid) for tid in st.session_state.gen_token_ids]
        decoded_text = tokenizer.decode(st.session_state.gen_token_ids[1:])
        st.markdown(f"**Raw Tokens sequence:** `{' | '.join(token_strs_gen)}`")
        st.success(f"**Generated Text:** {decoded_text if decoded_text else '[Empty - start generating!]'}")
        
        st.divider()
        if len(st.session_state.gen_token_ids) < max_seq_len:
            current_ids = torch.tensor([st.session_state.gen_token_ids], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                logits = model(current_ids)
                next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, k=5)
            
            candidates_data = [{"Token": tokenizer.id_to_token(tid.item()), "Token ID": tid.item(), "Probability (%)": round(prob.item() * 100, 2)} for tid, prob in zip(top_ids, top_probs)]
            df_candidates = pd.DataFrame(candidates_data)
            
            cols_candidates = st.columns(5)
            for i, cand in enumerate(candidates_data):
                with cols_candidates[i]:
                    if st.button(f"{cand['Token']}\n({cand['Probability (%)']}%)", key=f"cand_{cand['Token ID']}_{i}"):
                        st.session_state.gen_token_ids.append(cand['Token ID'])
                        st.rerun()

            if HAS_PLOTLY:
                fig_candidates = px.bar(df_candidates, x="Probability (%)", y="Token", orientation="h", color="Probability (%)", color_continuous_scale="Purples")
                fig_candidates.update_layout(yaxis={"categoryorder": "total ascending"}, template="plotly_dark", height=250)
                st.plotly_chart(fig_candidates, use_container_width=True)
        else:
            st.info("Maximum sequence length reached. Please reset to continue generating.")
