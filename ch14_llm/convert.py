#!/usr/bin/env python3
# 将 HuggingFace Qwen3-0.6B-Base (safetensors, bf16) 转换为 ch14_llm 引擎能直接加载的
# ncnn 自定义格式（Embed / RMSNorm / Gemm / InnerProduct 自定义层，blob 以 fp16 tag 存储）。
# 目标目录结构：<out>/text_embed.ncnn.{param,bin}, decoder_%02d_{pre,o,mlp}.ncnn.*,
#               decoder_norm.ncnn.*, lm_head.ncnn.*, vocab.txt, merges.txt
import json, struct, os, sys
import numpy as np

# 跨平台默认路径：优先命令行参数，其次环境变量 VULKAN_NN_MODEL_DIR，
# 最后回退到 Windows 专属路径（Linux 上请通过参数或环境变量指定）。
_MODEL_ROOT = os.environ.get("VULKAN_NN_MODEL_DIR", "F:/vulkan/model")
SRC = (sys.argv[1] if len(sys.argv) > 1
       else os.path.join(_MODEL_ROOT, "qwen3_0.6b_base_model.safetensors"))
OUT = (sys.argv[2] if len(sys.argv) > 2
       else os.path.join(_MODEL_ROOT, "qwen3_0.6b_base"))
VOCAB = 151936
HIDDEN = 1024
NL = 28

FP16_TAG = b'\x47\x6b\x30\x01'  # 0x01306B47 little-endian

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

def write_fp16(f, arr_f32):
    f16 = arr_f32.astype(np.float16)
    raw = f16.tobytes()
    f.write(FP16_TAG)
    padded = (len(raw) + 3) & ~3
    if len(raw) < padded:
        raw = raw + b'\x00' * (padded - len(raw))
    f.write(raw)

def write_param(path, body_lines):
    with open(path, 'w') as f:
        f.write("7767517\n")
        # layer_count blob_count（loader 仅用 layer_count；这里给个保守值）
        f.write(f"{len(body_lines)} 999\n")
        for ln in body_lines:
            f.write(ln + "\n")

def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"加载 {SRC} ...")
    T = load_safetensors(SRC)
    print(f"  tensors: {len(T)}")

    def get(name):
        return to_f32(*T[name])

    # text_embed + lm_head（tie_word_embeddings=true，二者共享）
    embed = get("model.embed_tokens.weight")   # [V, H]
    assert embed.shape == (VOCAB, HIDDEN), embed.shape

    # text_embed
    with open(f"{OUT}/text_embed.ncnn.bin", 'wb') as f:
        write_fp16(f, embed)
    write_param(f"{OUT}/text_embed.ncnn.param", [
        "Input                    in0                      0 1 in0",
        f"Embed                    embed_0                  1 1 in0 out0 0={HIDDEN} 1={VOCAB} 2=0 3={VOCAB*HIDDEN}",
    ])

    # lm_head（与 embed 相同权重）
    with open(f"{OUT}/lm_head.ncnn.bin", 'wb') as f:
        write_fp16(f, embed)
    write_param(f"{OUT}/lm_head.ncnn.param", [
        "Input        lm_in   0 1 lm_in",
        f"InnerProduct lm_ip   1 1 lm_in lm_out 0={VOCAB} 1=0 2={VOCAB*HIDDEN}",
    ])

    # decoder layers
    for i in range(NL):
        p = f"model.layers.{i}."
        g_pre = get(p + "input_layernorm.weight")
        q = get(p + "self_attn.q_proj.weight")     # [2048,1024]
        k = get(p + "self_attn.k_proj.weight")     # [1024,1024]
        v = get(p + "self_attn.v_proj.weight")     # [1024,1024]
        q_norm = get(p + "self_attn.q_norm.weight") # [128] Qwen3 QK-Norm
        k_norm = get(p + "self_attn.k_norm.weight") # [128]
        o = get(p + "self_attn.o_proj.weight")     # [1024,2048]
        g_mlp = get(p + "post_attention_layernorm.weight")
        gate = get(p + "mlp.gate_proj.weight")     # [3072,1024]
        up = get(p + "mlp.up_proj.weight")         # [3072,1024]
        down = get(p + "mlp.down_proj.weight")     # [1024,3072]

        # pre: rmsn_3(g_pre) + q_norm + k_norm + gemm_0(q) + gemm_1(k) + gemm_2(v)
        with open(f"{OUT}/decoder_{i:02d}_pre.ncnn.bin", 'wb') as f:
            write_fp16(f, g_pre); write_fp16(f, q_norm); write_fp16(f, k_norm)
            write_fp16(f, q); write_fp16(f, k); write_fp16(f, v)
        write_param(f"{OUT}/decoder_{i:02d}_pre.ncnn.param", [
            "Input                    in0                      0 1 in0",
            f"RMSNorm                  rmsn_3                   1 1 in0 1 0={HIDDEN} 1=1.000000e-6 2=0",
            "RMSNorm                  q_norm                   1 1 in0 1 0=128 1=1.000000e-6 2=0",
            "RMSNorm                  k_norm                   1 1 in0 1 0=128 1=1.000000e-6 2=0",
            "Split                    splitncnn_0              1 3 1 2 3 4",
            f"Gemm                     gemm_0                   1 1 4 5 10=-1 2=0 3=1 4=0 5=1 6=1 7=1 8={q.shape[0]} 9={q.shape[1]}",
            "Reshape                  reshape_7                1 1 5 6 0=128 1=16 2=1",
            "Permute                  permute_4                1 1 6 out0 0=2",
            f"Gemm                     gemm_1                   1 1 3 8 10=-1 2=0 3=1 4=0 5=1 6=1 7=1 8={k.shape[0]} 9={k.shape[1]}",
            "Reshape                  reshape_8                1 1 8 9 0=128 1=8 2=1",
            "Permute                  permute_5                1 1 9 out1 0=2",
            f"Gemm                     gemm_2                   1 1 2 11 10=-1 2=0 3=1 4=0 5=1 6=1 7=1 8={v.shape[0]} 9={v.shape[1]}",
            "Reshape                  reshape_9                1 1 11 12 0=128 1=8 2=1",
            "Permute                  permute_6                1 1 12 out2 0=2",
        ])

        # o: gemm_0(o)
        with open(f"{OUT}/decoder_{i:02d}_o.ncnn.bin", 'wb') as f:
            write_fp16(f, o)
        write_param(f"{OUT}/decoder_{i:02d}_o.ncnn.param", [
            "Input                    in0                      0 1 in0",
            f"Gemm                     gemm_0                   1 1 in0 out0 10=-1 2=0 3=1 4=0 5=1 6=1 7=1 8={o.shape[0]} 9={o.shape[1]}",
        ])

        # mlp: rmsn_3(g_mlp) + gemm_0(gate) + gemm_1(up) + gemm_2(down)
        with open(f"{OUT}/decoder_{i:02d}_mlp.ncnn.bin", 'wb') as f:
            write_fp16(f, g_mlp); write_fp16(f, gate); write_fp16(f, up); write_fp16(f, down)
        write_param(f"{OUT}/decoder_{i:02d}_mlp.ncnn.param", [
            "Input                    in0                      0 1 in0",
            f"RMSNorm                  rmsn_3                   1 1 in0 1 0={HIDDEN} 1=1.000000e-6 2=0",
            "Split                    splitncnn_0              1 2 1 2 3",
            f"Gemm                     gemm_0                   1 1 3 4 10=-1 2=0 3=1 4=0 5=1 6=1 7=1 8={gate.shape[0]} 9={gate.shape[1]}",
            f"Gemm                     gemm_1                   1 1 2 11 10=-1 2=0 3=1 4=0 5=1 6=1 7=1 8={up.shape[0]} 9={up.shape[1]}",
            f"Gemm                     gemm_2                   1 1 4 13 10=-1 2=0 3=1 4=0 5=1 6=1 7=1 8={down.shape[0]} 9={down.shape[1]}",
        ])

    # decoder_norm
    fn = get("model.norm.weight")
    with open(f"{OUT}/decoder_norm.ncnn.bin", 'wb') as f:
        write_fp16(f, fn)
    write_param(f"{OUT}/decoder_norm.ncnn.param", [
        "Input                    in0                      0 1 in0",
        f"RMSNorm                  rmsn_0                   1 1 in0 out0 0={HIDDEN} 1=1.000000e-6 2=0",
    ])

    # 复用 qwen3_asr 的 tokenizer（同族 Qwen BPE，vocab 对齐）
    src_tok = os.path.join(_MODEL_ROOT, "qwen3_asr_0.6b")
    for fn2 in ("vocab.txt", "merges.txt"):
        try:
            data = open(f"{src_tok}/{fn2}", 'rb').read()
            open(f"{OUT}/{fn2}", 'wb').write(data)
            print(f"  复制 tokenizer: {fn2} ({len(data)} bytes)")
        except FileNotFoundError:
            print(f"  警告: 缺少 {fn2}，需另行提供")

    print(f"转换完成 -> {OUT}")
    print(f"  embed.RMS={np.sqrt(np.mean(embed**2)):.4f}  fn.RMS={np.sqrt(np.mean(fn**2)):.4f}")

if __name__ == "__main__":
    main()
