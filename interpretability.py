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

    matplotlib.use(
        "Agg"
    )  # Prevents segmentation faults on macOS by using a non-GUI backend
    import matplotlib.pyplot as plt
    import seaborn as sns

# Ensure the local directory is in the path so we can import tinyllm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import TinyLLM, apply_rotary_emb
except ImportError:
    st.error(
        "Could not import `TinyLLM` from `tinyllm.py`. Please make sure `tinyllm.py` is in the same directory."
    )
    st.stop()

# Global config to control Head Ablation inside Attention
ablation_mask = {}  # key: (layer_idx, head_idx) -> bool (True if ablated)


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
        # SwiGLU activation formula: (SiLU(x * W1) * (x * W3))
        # This gives us the 512 intermediate neuron firing intensities
        acts = F.silu(self.w1(x)) * self.w3(x)  # [bsz, seqlen, hidden_dim]
        self.last_activations = acts.detach().cpu()
        return self.w2(acts)


# -------------------------------------------------------------
# 3. Model & Tokenizer Loader
# -------------------------------------------------------------
@st.cache_resource
def load_model_and_tokenizer(model_path, tokenizer_path):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    model = TinyLLM(
        vocab_size=vocab_size,
        dim=128,
        n_layers=4,
        n_heads=4,
        ffn_dim=512,
        max_seq_len=64,
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Hook attention and Feed-Forward layers
    for idx, block in enumerate(model.layers):
        block.attention = HookedAttention(block.attention, layer_idx=idx)
        block.feed_forward = HookedFeedForward(block.feed_forward, layer_idx=idx)

    return model, tokenizer


# -------------------------------------------------------------
# 4. Logit Lens & Attribution Computation Helper
# -------------------------------------------------------------
def run_logit_lens_and_attribution(model, tokenizer, tokens):
    """
    Passes tokens through the model layer-by-layer, collects the output deltas (residual additions)
    from each block, and projects these intermediate state deltas directly to logits.
    """
    bsz, seqlen = tokens.shape
    h = model.tok_embeddings(tokens)
    freqs_cis = model.freqs_cis[:seqlen].to(tokens.device)

    mask = None
    if seqlen > 1:
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)

    layer_names = ["Embedding Layer"]
    layer_states = []

    # Save base embedding state
    layer_states.append(h)

    # Pass through blocks, saving states for attribution
    curr_h = h
    for idx, layer in enumerate(model.layers):
        curr_h = layer(curr_h, freqs_cis, mask)
        layer_names.append(f"Transformer Layer {idx}")
        layer_states.append(curr_h)

    # Project states to Predictions
    predictions = []
    for state in layer_states:
        logits = model.output(model.norm(state))[0].detach()  # [seq_len, vocab_size]
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=1, dim=-1)

        layer_preds = []
        for seq_i in range(seqlen):
            tid = top_ids[seq_i].item()
            prob = top_probs[seq_i].item()
            layer_preds.append((tokenizer.id_to_token(tid), prob))
        predictions.append(layer_preds)

    # Calculate Layer-by-Layer Logit Contributions (Attribution)
    final_state = layer_states[-1]
    final_normed = model.norm(final_state)
    final_logits = model.output(final_normed)[
        0, -1
    ]  # Logits at last sequence index [vocab_size]

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
    """
    Computes gradients of the top predicted token logit with respect to the input embeddings.
    Allows highlighting of which words causal influence was strongest.
    """
    model.zero_grad()

    embeddings = model.tok_embeddings.weight
    bsz, seqlen = tokens.shape

    embeddings_grad = embeddings.clone().detach().requires_grad_(True)
    h = embeddings_grad[tokens[0]]
    h = h.unsqueeze(0)

    freqs_cis = model.freqs_cis[:seqlen].to(tokens.device)
    mask = None
    if seqlen > 1:
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)

    # Forward pass
    curr_h = h
    for layer in model.layers:
        curr_h = layer(curr_h, freqs_cis, mask)
    curr_h = model.norm(curr_h)
    logits = model.output(curr_h)

    last_position_logits = logits[0, -1, :]
    top_class_id = torch.argmax(last_position_logits).item()
    target_score = last_position_logits[top_class_id]

    # Backward pass
    target_score.backward()

    grad_at_embeddings = embeddings_grad.grad
    if grad_at_embeddings is not None:
        position_saliency = (
            torch.norm(grad_at_embeddings[tokens[0]], dim=-1).cpu().numpy()
        )
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

# Custom premium styling
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
st.write(
    "Explore how your trained TinyLLM model represents and processes the Korean language."
)

# Sidebar Configuration
st.sidebar.header("📁 Model Configurations")
model_file = st.sidebar.text_input("Model Weights Path", "tiny_llm.pth")
tokenizer_file = st.sidebar.text_input("Tokenizer Path", "tokenizer.json")

if not os.path.exists(model_file) or not os.path.exists(tokenizer_file):
    st.sidebar.error(
        "⚠️ Model file or Tokenizer file not found. Please verify the absolute paths."
    )
    st.stop()
else:
    model, tokenizer = load_model_and_tokenizer(model_file, tokenizer_file)
    st.sidebar.success("✅ Model and Tokenizer loaded successfully!")

# Head Ablation Settings on Sidebar
st.sidebar.divider()
st.sidebar.subheader("✂️ Attention Head Ablation")
st.sidebar.write(
    "Toggle specific heads to ablate (turn off) them inside the model dynamically during calculation."
)

# Build active ablation grid interface
for l_i in range(4):
    cols_heads = st.sidebar.columns(4)
    for h_i in range(4):
        with cols_heads[h_i]:
            key = f"L{l_i}H{h_i}"
            is_checked = st.checkbox(
                key, value=False, help=f"Ablate Layer {l_i} Head {h_i}"
            )
            ablation_mask[(l_i, h_i)] = is_checked

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

# -------------------------------------------------------------
# Tab 1: Self-Attention Mapping
# -------------------------------------------------------------
with tab1:
    st.header("Attention Maps Visualization")
    st.write(
        "Visualize the self-attention weights inside the transformer layers to see which words route information to each other."
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        test_sentence = st.text_input(
            "Enter a sentence to analyze:",
            "오늘 날씨가 아주 좋습니다",
            key="tab1_input",
        )

        layer_selection = st.selectbox(
            "Select Layer:", [0, 1, 2, 3], index=3, key="tab1_layer"
        )
        head_selection = st.selectbox(
            "Select Attention Head:", [0, 1, 2, 3], index=0, key="tab1_head"
        )

        cls_id = tokenizer.token_to_id("[CLS]")
        tokens = [cls_id] + tokenizer.encode(test_sentence).ids
        token_strs = [tokenizer.id_to_token(t) for t in tokens]

    with col2:
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            model(input_tensor)

        block = model.layers[layer_selection]
        attn_matrix = block.attention.last_attention_weights[0, head_selection].numpy()

        if HAS_PLOTLY:
            fig = go.Figure(
                data=go.Heatmap(
                    z=attn_matrix,
                    x=token_strs,
                    y=token_strs,
                    colorscale="Cividis",
                    colorbar=dict(title="Attention"),
                )
            )
            fig.update_layout(
                title=f"Attention Map (Layer {layer_selection}, Head {head_selection})",
                xaxis_title="Key Tokens (Attended To)",
                yaxis_title="Query Tokens (Attending)",
                template="plotly_dark",
                width=700,
                height=550,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                attn_matrix,
                xticklabels=token_strs,
                yticklabels=token_strs,
                cmap="cividis",
                annot=True,
                fmt=".2f",
                ax=ax,
            )
            st.pyplot(fig)

# -------------------------------------------------------------
# Tab 2: Logit Lens & Attribution Breakdown
# -------------------------------------------------------------
with tab2:
    st.header("The Logit Lens & Residual Attribution")
    st.write(
        "Inspect how predictions build up layer-by-layer and analyze which layers contributed the largest logit boosts to the final token choice."
    )

    lens_sentence = st.text_input(
        "Enter a sentence to pass through the lens:",
        "오늘 날씨가 아주 좋습니다",
        key="tab2_lens_input",
    )

    cls_id = tokenizer.token_to_id("[CLS]")
    lens_tokens = [cls_id] + tokenizer.encode(lens_sentence).ids
    lens_token_strs = [tokenizer.id_to_token(t) for t in lens_tokens]

    input_tensor = torch.tensor([lens_tokens], dtype=torch.long)

    layer_names, predictions, final_token, attribs = run_logit_lens_and_attribution(
        model, tokenizer, input_tensor
    )

    st.subheader(f"1. Predictions Grid (Final Token Predicted: `{final_token}`)")

    def render_logit_lens_html(layer_names, token_strs, predictions):
        html = "<table style='width:100%; border-collapse: collapse; color: white; background-color: #1F2937;'>"
        html += "<tr style='border-bottom: 2px solid #4F46E5;'>"
        html += "<th style='padding: 12px; text-align: left; background-color: #111827;'>Layer</th>"
        for token in token_strs:
            html += f"<th style='padding: 12px; text-align: center; background-color: #111827;'>{token}</th>"
        html += "</tr>"

        for l_idx, layer_name in enumerate(layer_names):
            html += "<tr style='border-bottom: 1px solid #374151;'>"
            html += f"<td style='padding: 12px; font-weight: bold; background-color: #111827;'>{layer_name}</td>"
            for col_idx in range(len(token_strs)):
                word, prob = predictions[l_idx][col_idx]
                opacity = min(max(prob, 0.05), 1.0)
                bg_color = f"rgba(79, 70, 229, {opacity})"
                html += f"<td style='padding: 12px; text-align: center; background-color: {bg_color}; border: 1px solid #374151;'>"
                html += f"<div style='font-weight: bold; font-size: 14px;'>{word}</div>"
                html += f"<div style='font-size: 11px; opacity: 0.8;'>{prob * 100:.1f}%</div>"
                html += "</td>"
            html += "</tr>"
        html += "</table>"
        return html

    st.markdown(
        render_logit_lens_html(layer_names, lens_token_strs, predictions),
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader(f"2. Layer-by-Layer Attribution for Predicted Token: `{final_token}`")

    df_attribs = pd.DataFrame(attribs)

    col_chart, col_explain = st.columns([2, 1])

    with col_chart:
        if HAS_PLOTLY:
            fig_waterfall = go.Figure(
                go.Waterfall(
                    name="Attribution",
                    orientation="v",
                    measure=["relative"] * len(df_attribs),
                    x=df_attribs["Layer"],
                    textposition="outside",
                    text=df_attribs["Delta Contribution (Logit Units)"].astype(str),
                    y=df_attribs["Delta Contribution (Logit Units)"],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                )
            )
            fig_waterfall.update_layout(
                title=f"Logit Contribution Breakdown (Why did model choose '{final_token}'?)",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(
                data=df_attribs,
                x="Layer",
                y="Delta Contribution (Logit Units)",
                palette="coolwarm",
                ax=ax,
            )
            st.pyplot(fig)

    with col_explain:
        st.write("🔬 **What is Layer Attribution?**")
        st.markdown(f"""
        Since representations are added sequentially down the residual stream, we can measure how much each layer boosts or suppresses the final logit score for **`{final_token}`**.
        * **Positive Delta**: The layer added vectors that correspond to this token (reinforcement).
        * **Negative Delta**: The layer pushed the model away from this token (error-correction).
        """)
        st.dataframe(df_attribs)

# -------------------------------------------------------------
# Tab 3: LLM fMRI (Neuron Activations)
# -------------------------------------------------------------
with tab3:
    st.header("🧠 LLM fMRI: Neuron Activation Firing")
    st.write(
        "In standard brains, different regions light up when exposed to different words. Similarly, we can track the activation profile of the 512 hidden units (neurons) in the SwiGLU Feed-Forward networks."
    )

    fmri_sentence = st.text_input(
        "Enter a sentence for fMRI scan:",
        "오늘 날씨가 아주 좋습니다",
        key="tab3_fmri_input",
    )

    cls_id = tokenizer.token_to_id("[CLS]")
    fmri_tokens = [cls_id] + tokenizer.encode(fmri_sentence).ids
    fmri_token_strs = [tokenizer.id_to_token(t) for t in fmri_tokens]

    input_tensor = torch.tensor([fmri_tokens], dtype=torch.long)

    # Run forward pass to compute activations
    with torch.no_grad():
        model(input_tensor)

    fmri_layer_idx = st.selectbox(
        "Select Layer for fMRI scan:", [0, 1, 2, 3], index=2, key="tab3_fmri_layer"
    )

    # Extract the activations from the HookedFeedForward wrapper
    block = model.layers[fmri_layer_idx]
    activations = block.feed_forward.last_activations[0].numpy()  # [seqlen, 512]

    st.subheader("1. Active Neuron Heatmap Grid")
    st.write(
        "Visualizes the activation levels of neurons across the sentence. Select the density slider to filter down to the most active neurons."
    )

    num_neurons_to_show = st.slider("Number of active neurons to display:", 10, 100, 30)

    # Identify which neurons had the highest peak activations across the sentence
    peak_activations = activations.max(axis=0)  # [512]
    top_neuron_indices = np.argsort(peak_activations)[::-1][:num_neurons_to_show]

    # Filter activation matrix to only show these top neurons
    filtered_activations = activations[
        :, top_neuron_indices
    ].T  # [num_neurons_to_show, seqlen]
    neuron_labels = [f"Neuron {idx}" for idx in top_neuron_indices]

    if HAS_PLOTLY:
        fig_fmri = go.Figure(
            data=go.Heatmap(
                z=filtered_activations,
                x=fmri_token_strs,
                y=neuron_labels,
                colorscale="Hot",
                colorbar=dict(title="Activation"),
            )
        )
        fig_fmri.update_layout(
            title=f"fMRI Brain Scan of FFN layer {fmri_layer_idx} (Top {num_neurons_to_show} Neurons)",
            xaxis_title="Input Sentence Tokens",
            yaxis_title="Neuron Index",
            template="plotly_dark",
            height=200 + num_neurons_to_show * 15,
        )
        st.plotly_chart(fig_fmri, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            filtered_activations,
            xticklabels=fmri_token_strs,
            yticklabels=neuron_labels,
            cmap="hot",
            ax=ax,
        )
        st.pyplot(fig)

    st.divider()

    col_tok, col_neu = st.columns(2)

    with col_tok:
        st.subheader("🔍 Token Firing Analysis")
        st.write(
            "Select a word in the sentence to see which specific neurons fired (activated) the most for it:"
        )
        selected_token = st.selectbox(
            "Select Word:", fmri_token_strs, index=min(1, len(fmri_token_strs) - 1)
        )

        token_pos = fmri_token_strs.index(selected_token)
        token_acts = activations[token_pos]  # [512]

        # Get top 10 neurons that fired for this token
        top_token_neurons = np.argsort(token_acts)[::-1][:10]
        top_token_vals = token_acts[top_token_neurons]

        df_tok_neu = pd.DataFrame(
            {
                "Neuron": [f"Neuron #{idx}" for idx in top_token_neurons],
                "Activation Value": np.round(top_token_vals, 3),
            }
        )

        if HAS_PLOTLY:
            fig_tok_neu = px.bar(
                df_tok_neu,
                x="Activation Value",
                y="Neuron",
                orientation="h",
                title=f"Top 10 Neurons Firing for '{selected_token}'",
                color="Activation Value",
                color_continuous_scale="Reds",
            )
            fig_tok_neu.update_layout(
                yaxis={"categoryorder": "total ascending"},
                template="plotly_dark",
                height=300,
            )
            st.plotly_chart(fig_tok_neu, use_container_width=True)
        else:
            st.dataframe(df_tok_neu)

    with col_neu:
        st.subheader("🔬 Single Neuron Profiler")
        st.write(
            "Analyze a specific neuron to see its activation level across all words in the sentence (acts as a specialized feature detector)."
        )

        neuron_id = st.number_input(
            "Enter Neuron ID (0 to 511):",
            min_value=0,
            max_value=511,
            value=int(top_neuron_indices[0]),
        )

        neuron_acts_across_sentence = activations[:, neuron_id]  # [seqlen]

        df_neu_acts = pd.DataFrame(
            {
                "Token": fmri_token_strs,
                "Activation Firing": np.round(neuron_acts_across_sentence, 3),
            }
        )

        if HAS_PLOTLY:
            fig_neu_acts = px.bar(
                df_neu_acts,
                x="Token",
                y="Activation Firing",
                title=f"Neuron #{neuron_id} Firing Pattern Across Sentence",
                color="Activation Firing",
                color_continuous_scale="Oranges",
            )
            fig_neu_acts.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_neu_acts, use_container_width=True)
        else:
            st.dataframe(df_neu_acts)

# -------------------------------------------------------------
# Tab 4: Causal Word Saliency (Gradients)
# -------------------------------------------------------------
with tab4:
    st.header("Causal Word Saliency Mapping")
    st.write(
        "Find out which words in the context had the strongest causal influence on predicting the final token using backpropagated gradients."
    )

    saliency_sentence = st.text_input(
        "Enter a sentence to analyze:", "오늘 날씨가 아주 좋습니다", key="tab4_input"
    )

    cls_id = tokenizer.token_to_id("[CLS]")
    saliency_tokens = [cls_id] + tokenizer.encode(saliency_sentence).ids
    saliency_token_strs = [tokenizer.id_to_token(t) for t in saliency_tokens]

    input_tensor = torch.tensor([saliency_tokens], dtype=torch.long)

    # Calculate gradients
    normalized_saliency = run_gradient_saliency(model, tokenizer, input_tensor)

    # Predict the target output class to explain
    with torch.no_grad():
        final_preds = model(input_tensor)
    target_token_id = torch.argmax(final_preds[0, -1, :]).item()
    target_token_str = tokenizer.id_to_token(target_token_id)

    st.markdown(f"### Causal importance for predicting: **`{target_token_str}`**")
    st.write(
        "Words with darker blue background colors have a higher causal relationship to the prediction:"
    )

    # Generate styled HTML blocks
    html_saliency = "<div style='display: flex; gap: 8px; flex-wrap: wrap; padding: 20px; background-color: #111827; border-radius: 8px; border: 1px solid #374151;'>"
    for word_i, token_str in enumerate(saliency_token_strs):
        score = normalized_saliency[word_i]
        bg_color = f"rgba(79, 70, 229, {score * 0.9 + 0.1})"  # dynamic opacity
        html_saliency += f"<div style='padding: 8px 16px; background-color: {bg_color}; border-radius: 4px; border: 1px solid rgba(255,255,255,0.1); text-align: center; color: white;'>"
        html_saliency += (
            f"<div style='font-weight: bold; font-size: 16px;'>{token_str}</div>"
        )
        html_saliency += (
            f"<div style='font-size: 11px; opacity: 0.8;'>{score:.2f}</div>"
        )
        html_saliency += "</div>"
    html_saliency += "</div>"

    st.markdown(html_saliency, unsafe_allow_html=True)

    st.markdown("""
    💡 **How to interpret Saliency Maps:**
    * Saliency highlights the **sensitivity** of the output node to small changes in each token's vector representation.
    """)

# -------------------------------------------------------------
# Tab 5: Predictability & Entropy
# -------------------------------------------------------------
with tab5:
    st.header("Next-Token Predictability & Entropy Profile")
    st.write(
        "Understand where the model is confident versus where it is surprised (high information entropy)."
    )

    cls_id = tokenizer.token_to_id("[CLS]")
    tokens = [cls_id] + tokenizer.encode(test_sentence).ids
    token_strs = [tokenizer.id_to_token(t) for t in tokens]

    input_tensor = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)

    probs = F.softmax(logits[0], dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1).cpu().numpy()

    analysis_data = []
    for idx, token in enumerate(token_strs):
        if idx == len(token_strs) - 1:
            break
        next_token = token_strs[idx + 1]
        next_token_id = tokens[idx + 1]
        confidence = probs[idx, next_token_id].item()

        analysis_data.append(
            {
                "Position": idx,
                "Context Word": token,
                "Predicted Word": next_token,
                "Surprisal (Entropy in bits)": round(entropy[idx], 2),
                "Confidence Probability (%)": round(confidence * 100, 2),
            }
        )

    df_analysis = pd.DataFrame(analysis_data)

    col1, col2 = st.columns([2, 1])

    with col1:
        if HAS_PLOTLY:
            fig = px.line(
                df_analysis,
                x="Context Word",
                y="Surprisal (Entropy in bits)",
                markers=True,
                title="Uncertainty (Entropy) Profile across the Sentence",
                labels={"Surprisal (Entropy in bits)": "Uncertainty (Bits)"},
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(
                data=df_analysis,
                x="Context Word",
                y="Surprisal (Entropy in bits)",
                marker="o",
                ax=ax,
            )
            st.pyplot(fig)

    with col2:
        st.write("🔬 **How to Read the Entropy Profile:**")
        st.markdown("""
        * **High Entropy (Uncertainty)**: The model has many possible choices for the next token. This usually happens at the start of a sentence or phrase.
        * **Low Entropy (Confidence)**: The model is highly certain of the next token. This happens after context clues.
        """)

    st.dataframe(df_analysis, use_container_width=True)

# -------------------------------------------------------------
# Tab 6: Interactive Decoding Playground
# -------------------------------------------------------------
with tab6:
    st.header("Interactive Generation & Decoding Steering")
    st.write(
        "Type a prompt and watch the model generate tokens. You can click candidate tokens to override the model's choices (steering)."
    )

    # Manage session state for generated tokens
    cls_id = tokenizer.token_to_id("[CLS]")
    if "gen_token_ids" not in st.session_state:
        st.session_state.gen_token_ids = [cls_id]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Decoding Parameters")
        temperature = st.slider("Temperature (randomness):", 0.1, 2.0, 0.8, step=0.1)
        top_k_val = st.slider("Top-K filtering:", 1, 100, 50, step=1)

        # User prompt initializer
        custom_prompt = st.text_input("Reset with custom prompt:", "")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Apply Prompt & Reset"):
                if custom_prompt.strip():
                    st.session_state.gen_token_ids = [cls_id] + tokenizer.encode(
                        custom_prompt
                    ).ids
                else:
                    st.session_state.gen_token_ids = [cls_id]
                st.rerun()
        with col_btn2:
            if st.button("Reset to Empty [CLS]"):
                st.session_state.gen_token_ids = [cls_id]
                st.rerun()

        # Auto step generation
        if st.button("🤖 Auto-Generate 1 Token"):
            current_ids = torch.tensor(
                [st.session_state.gen_token_ids], dtype=torch.long
            )
            with torch.no_grad():
                logits = model(current_ids)
                next_token_logits = logits[0, -1, :] / temperature

                if top_k_val > 0:
                    top_k_threshold = torch.topk(next_token_logits, top_k_val)[0][
                        ..., -1, None
                    ]
                    indices_to_remove = next_token_logits < top_k_threshold
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                st.session_state.gen_token_ids.append(next_token)
                st.rerun()

    with col2:
        st.subheader("📝 Generated Output")

        # Decode and render sequence
        token_strs_gen = [
            tokenizer.id_to_token(tid) for tid in st.session_state.gen_token_ids
        ]
        decoded_text = tokenizer.decode(
            st.session_state.gen_token_ids[1:]
        )  # Skip [CLS]

        st.markdown(f"**Raw Tokens sequence:** `{' | '.join(token_strs_gen)}`")
        st.success(
            f"**Generated Text:** {decoded_text if decoded_text else '[Empty - start generating!]'}"
        )

        st.divider()
        st.subheader("🎯 Next Token Predictions (Top 5 Candidates)")

        # Calculate next token candidates for the current sequence
        current_ids = torch.tensor([st.session_state.gen_token_ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(current_ids)
            next_token_logits = logits[0, -1, :]

        probs = F.softmax(next_token_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=5)

        candidates_data = []
        for i in range(5):
            tid = top_ids[i].item()
            prob = top_probs[i].item()
            candidates_data.append(
                {
                    "Token": tokenizer.id_to_token(tid),
                    "Token ID": tid,
                    "Probability (%)": round(prob * 100, 2),
                }
            )

        df_candidates = pd.DataFrame(candidates_data)

        # Render candidate buttons for manual steering
        st.write(
            "Click a candidate token below to add it to the sequence (Steer the model):"
        )
        cols_candidates = st.columns(5)
        for i in range(5):
            candidate_token = candidates_data[i]["Token"]
            candidate_id = candidates_data[i]["Token ID"]
            candidate_prob = candidates_data[i]["Probability (%)"]

            with cols_candidates[i]:
                if st.button(
                    f"{candidate_token}\n({candidate_prob}%)",
                    key=f"candidate_{candidate_id}_{i}",
                ):
                    st.session_state.gen_token_ids.append(candidate_id)
                    st.rerun()

        # Visualize candidate probabilities
        if HAS_PLOTLY:
            fig_candidates = px.bar(
                df_candidates,
                x="Probability (%)",
                y="Token",
                orientation="h",
                color="Probability (%)",
                color_continuous_scale="Purples",
            )
            fig_candidates.update_layout(
                yaxis={"categoryorder": "total ascending"},
                template="plotly_dark",
                height=250,
            )
            st.plotly_chart(fig_candidates, use_container_width=True)
