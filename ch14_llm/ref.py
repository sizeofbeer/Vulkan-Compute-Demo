#!/usr/bin/env python3
# 独立 numpy 参考实现：读取与 C++ 完全相同的 ncnn 权重（同样 tag 逻辑、同样 blob 顺序），
# 但前向按 Qwen3 规范独立重写，用于核对 C++ 引擎数值是否正确。
import struct, sys, time, os, json
import numpy as np

_MODEL_ROOT = os.environ.get("VULKAN_NN_MODEL_DIR", "F:/vulkan/model")
MODEL = (os.environ.get("CH14_MODEL")
         or os.path.join(_MODEL_ROOT, "qwen3_asr_0.6b"))

class BinReader:
    def __init__(self, path):
        self.buf = open(path,'rb').read()
        self.off = 0
    def read(self, count):
        tag = self.buf[self.off:self.off+4]
        if tag == b'\x47\x6b\x30\x01':  # 0x01306B47 little-endian
            self.off += 4
            padded = (count*2 + 3) & ~3
            raw = self.buf[self.off:self.off+padded]; self.off += padded
            return np.frombuffer(raw[:count*2], dtype=np.float16).astype(np.float32)
        else:
            raw = self.buf[self.off:self.off+count*4]; self.off += count*4
            return np.frombuffer(raw, dtype=np.float32)

def load_param_file(dir, base):
    pf = open(f"{dir}/{base}.ncnn.param", encoding='utf-8').read().splitlines()
    layer_count, blob_count = map(int, pf[1].split())
    br = BinReader(f"{dir}/{base}.ncnn.bin")
    out = {}
    for li in range(layer_count):
        toks = pf[2+li].split()
        if len(toks) < 4: continue
        typ, name = toks[0], toks[1]
        inc, outc = int(toks[2]), int(toks[3])
        pm = {}
        for t in toks[4+inc+outc:]:
            if '=' in t:
                k,v = t.split('='); pm[int(k)] = float(v)
        if typ == "Embed":
            dim, vocab = int(pm[0]), int(pm[1])
            total = int(pm.get(3, dim*vocab))
            out[name] = br.read(total).reshape(vocab, dim)
        elif typ == "RMSNorm":
            dim = int(pm[0])
            out[name] = br.read(dim)
        elif typ == "Gemm":
            N, K = int(pm[8]), int(pm[9])
            out[name] = br.read(N*K).reshape(N, K)
        elif typ == "InnerProduct":
            N = int(pm[0]); total = int(pm[2])
            out[name] = br.read(total).reshape(N, total//N)
    return out

# 配置
VOCAB=151936; H=1024; NL=28; NQ=16; NKV=8; HD=128; INTER=3072; THETA=1e6; EPS=1e-6
GRP = NQ//NKV

def load_safetensors(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        data_start = 8 + n
        tensors = {}
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            dt = meta['dtype']; shape = meta['shape']
            s, e = meta['data_offsets']
            f.seek(data_start + s)
            raw = f.read(e - s)
            tensors[name] = (dt, shape, raw)
    return tensors

def to_f32(dt, shape, raw):
    if dt == 'BF16':
        u16 = np.frombuffer(raw, dtype=np.uint16)
        i32 = u16.astype(np.uint32) << 16
        return i32.view(np.float32).reshape(shape)
    elif dt == 'F16':
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    elif dt == 'F32':
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    else:
        raise ValueError(f"unsupported dtype {dt}")

def load_all_safetensors(path):
    """直接从原始 bf16 safetensors 加载（全 fp32），用于隔离 fp16 转换的影响。"""
    T = load_safetensors(path)
    def g(name): return to_f32(*T[name])
    W = {}
    W['embed'] = g("model.embed_tokens.weight")
    W['final'] = g("model.norm.weight")
    # tie：lm = embed
    W['lm'] = W['embed']
    layers = []
    for i in range(NL):
        p = f"model.layers.{i}."
        layers.append({
            'g_pre': g(p+"input_layernorm.weight"),
            'wq': g(p+"self_attn.q_proj.weight"),
            'wk': g(p+"self_attn.k_proj.weight"),
            'wv': g(p+"self_attn.v_proj.weight"),
            'q_norm': g(p+"self_attn.q_norm.weight"),
            'k_norm': g(p+"self_attn.k_norm.weight"),
            'wo': g(p+"self_attn.o_proj.weight"),
            'g_mlp': g(p+"post_attention_layernorm.weight"),
            'wg': g(p+"mlp.gate_proj.weight"),
            'wu': g(p+"mlp.up_proj.weight"),
            'wd': g(p+"mlp.down_proj.weight"),
        })
    W['layers'] = layers
    return W

def load_all():
    W = {}
    W['embed'] = load_param_file(MODEL, "text_embed")["embed_0"]
    W['final'] = load_param_file(MODEL, "decoder_norm")["rmsn_0"]
    W['lm']    = load_param_file(MODEL, "lm_head")["lm_ip"]
    W['layers'] = []
    for L in range(NL):
        pre = load_param_file(MODEL, f"decoder_{L:02d}_pre")
        o   = load_param_file(MODEL, f"decoder_{L:02d}_o")
        mlp = load_param_file(MODEL, f"decoder_{L:02d}_mlp")
        W['layers'].append({
            'g_pre':pre["rmsn_3"], 'wq':pre["gemm_0"], 'wk':pre["gemm_1"], 'wv':pre["gemm_2"],
            'q_norm':pre.get("q_norm"), 'k_norm':pre.get("k_norm"),
            'wo':o["gemm_0"],
            'g_mlp':mlp["rmsn_3"], 'wg':mlp["gemm_0"], 'wu':mlp["gemm_1"], 'wd':mlp["gemm_2"],
        })
    return W

def rmsnorm(x, gamma):
    ss = np.mean(x*x, axis=-1, keepdims=True)
    return x * (1.0/np.sqrt(ss+EPS)) * gamma

def qknorm(x, gamma, eps=1e-6):
    # x: [seq, n_heads, HD]; gamma: [HD]
    ss = np.mean(x*x, axis=-1, keepdims=True)
    return x * (1.0/np.sqrt(ss+eps)) * gamma

def rope(x, pos):
    half = HD//2
    inv = THETA ** (-2.0*np.arange(half)/HD)
    t = (pos[:,None]*inv[None,:])
    c = np.cos(t); s = np.sin(t)
    c = np.concatenate([c,c],axis=-1)
    s = np.concatenate([s,s],axis=-1)
    x0 = x[..., :half]; x1 = x[..., half:]
    c = c[:,None,:]; s = s[:,None,:]   # [seq,1,HD] 便于与 [seq,n_heads,HD] 广播
    o0 = x0*c[...,:half] - x1*s[...,:half]
    o1 = x0*s[...,:half] + x1*c[...,:half]
    return np.concatenate([o0,o1],axis=-1)

def forward(tokens, W):
    embed = W['embed']
    x = embed[tokens].astype(np.float32)
    seq = len(tokens)
    kv_k = [None]*NL; kv_v=[None]*NL
    for L in range(NL):
        w = W['layers'][L]
        g_pre=w['g_pre']; wq=w['wq']; wk=w['wk']; wv=w['wv']; wo=w['wo']
        g_mlp=w['g_mlp']; wg=w['wg']; wu=w['wu']; wd=w['wd']
        hn = rmsnorm(x, g_pre)
        q = (hn @ wq.T).reshape(seq, NQ, HD)
        k = (hn @ wk.T).reshape(seq, NKV, HD)
        v = (hn @ wv.T).reshape(seq, NKV, HD)
        # Qwen3 QK-Norm（逐头 RMSNorm，RoPE 之前）
        if w.get('q_norm') is not None:
            q = qknorm(q, w['q_norm'])
            k = qknorm(k, w['k_norm'])
        q = rope(q, np.arange(seq)); k = rope(k, np.arange(seq))
        if kv_k[L] is None:
            kv_k[L]=k; kv_v[L]=v
        else:
            kv_k[L]=np.concatenate([kv_k[L],k],axis=0); kv_v[L]=np.concatenate([kv_v[L],v],axis=0)
        Kc=kv_k[L]; Vc=kv_v[L]; T=Kc.shape[0]
        out = np.zeros((seq, NQ, HD), dtype=np.float32)
        scale = 1.0/np.sqrt(HD)
        for qi in range(seq):
            for h in range(NQ):
                g = h//GRP
                qq = q[qi,h]; kk = Kc[:qi+1, g, :]; sc = (kk @ qq)*scale
                sc = sc - sc.max(); e = np.exp(sc); a = e/e.sum()
                vv = Vc[:qi+1, g, :]; out[qi,h] = a @ vv
        attn = out.reshape(seq, NQ*HD)
        ao = attn @ wo.T
        x = x + ao
        hn2 = rmsnorm(x, g_mlp)
        gate = hn2 @ wg.T; up = hn2 @ wu.T
        act = gate/(1.0+np.exp(-gate))*up
        down = act @ wd.T
        x = x + down
    xlast = rmsnorm(x[seq-1:seq], W['final'])[0]
    logits = xlast @ W['lm'].T
    return logits

if __name__ == "__main__":
    raw = os.environ.get("CH14_RAW")
    t0=time.time()
    if raw:
        W = load_all_safetensors(raw)
        src = f"RAW safetensors {raw}"
    else:
        W = load_all()
        src = f"ncnn {MODEL}"
    print(f"[ref] 权重加载({src}) {time.time()-t0:.2f}s  embed.RMS={np.sqrt(np.mean(W['embed']**2)):.4f}  lm.RMS={np.sqrt(np.mean(W['lm']**2)):.4f}  final.RMS={np.sqrt(np.mean(W['final']**2)):.4f}")
    ids = [785, 6722, 315, 9625, 374]
    t1=time.time()
    logits = forward(ids, W)
    print(f"[ref] forward 耗时 {time.time()-t1:.2f}s")
    print(f"[ref] logits RMS={np.sqrt(np.mean(logits**2)):.3f}  max={logits.max():.3f}")
    top = np.argsort(-logits)[:5]
    for i in top:
        print(f"   {i}  logit={logits[i]:.3f}")
    # 自回归几步
    cur = ids[:]
    print("[ref] greedy:", end=" ")
    for _ in range(8):
        lg = forward(cur, W); nxt=int(np.argmax(lg)); cur.append(nxt); print(nxt, end=" ")
    print()
