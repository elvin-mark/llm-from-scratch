"""
TinyLLM inference in pure NumPy (Hyper-Compact).
"""
import argparse
import numpy as np

def apply_rotary_emb(x, fc):
    xc = x.reshape(*x.shape[:-1], -1, 2)
    xc = xc[..., 0] + 1j * xc[..., 1]
    out = xc * fc[None, :, None, :]
    return np.stack([out.real, out.imag], axis=-1).reshape(x.shape).astype(np.float32)

def generate(w, encode, decode, prompt, max_new=40, temp=0.8, dim=128, n_heads=4, max_seq=128):
    tokens = list(encode(prompt))
    for _ in range(max_new):
        b, s = 1, min(len(tokens), max_seq)
        window = tokens[-max_seq:]
        h = w["tok_embeddings.weight"][window][None, ...]
        
        fc = np.exp(1j * np.outer(np.arange(s), 1.0 / (10000.0 ** (np.arange(0, dim // n_heads, 2) / (dim // n_heads)))))
        mask = np.triu(np.full((s, s), -np.inf, dtype=np.float32), k=1)[None, None, :, :] if s > 1 else 0
        
        l = 0
        while f"layers.{l}.attention.wq.weight" in w:
            pfx = f"layers.{l}"
            # RMSNorm + QKV Projection
            x = h * (1.0 / np.sqrt(np.mean(h**2, axis=-1, keepdims=True) + 1e-6)) * w[f"{pfx}.attention_norm.weight"]
            xq, xk, xv = [x @ w[f"{pfx}.attention.w{c}.weight"].T for c in "qkv"]
            
            # RoPE
            xq = apply_rotary_emb(xq.reshape(b, s, n_heads, -1), fc).transpose(0, 2, 1, 3)
            xk = apply_rotary_emb(xk.reshape(b, s, n_heads, -1), fc).transpose(0, 2, 1, 3)
            xv = xv.reshape(b, s, n_heads, -1).transpose(0, 2, 1, 3)
            
            # Causal Self-Attention
            sc = xq @ xk.transpose(0, 1, 3, 2) / np.sqrt(dim // n_heads) + mask
            pr = np.exp(sc - np.max(sc, axis=-1, keepdims=True))
            pr = pr / np.sum(pr, axis=-1, keepdims=True)
            attn = (pr @ xv).transpose(0, 2, 1, 3).reshape(b, s, -1)
            h = h + attn @ w[f"{pfx}.attention.wo.weight"].T
            
            # RMSNorm + SwiGLU FFN
            x = h * (1.0 / np.sqrt(np.mean(h**2, axis=-1, keepdims=True) + 1e-6)) * w[f"{pfx}.ffn_norm.weight"]
            h1 = x @ w[f"{pfx}.feed_forward.w1.weight"].T
            h = h + ((h1 / (1.0 + np.exp(-h1))) * (x @ w[f"{pfx}.feed_forward.w3.weight"].T)) @ w[f"{pfx}.feed_forward.w2.weight"].T
            l += 1
            
        h = h * (1.0 / np.sqrt(np.mean(h**2, axis=-1, keepdims=True) + 1e-6)) * w["norm.weight"]
        logits = (h @ w["output.weight"].T)[0, -1]
        
        if temp <= 0: tokens.append(int(np.argmax(logits)))
        else:
            p = np.exp(logits / temp - np.max(logits / temp))
            tokens.append(int(np.random.choice(len(p), p=p / np.sum(p))))
            
    return decode(tokens)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TinyLLM inference in pure NumPy")
    p.add_argument("--weights", required=True, help="Path to .pth or .npz weights")
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()
    
    if args.weights.endswith(".npz"):
        w = dict(np.load(args.weights))
    else:
        import torch
        w = {k: v.detach().numpy().astype(np.float32) for k, v in torch.load(args.weights, map_location="cpu").items()}
        
    import os
    from tokenizers import Tokenizer
    tokenizer_file = "checkpoints/tokenizer.json" if os.path.exists("checkpoints/tokenizer.json") else "tokenizer.json"
    tok = Tokenizer.from_file(tokenizer_file)
    print(generate(w, lambda t: tok.encode(t).ids, tok.decode, args.prompt, args.tokens, args.temperature, args.dim, args.n_heads, args.max_seq_len))
