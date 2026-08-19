#!/usr/bin/env python3
# 校验 C++ 引擎 dump 的张量与参考实现 dump 的张量是否一致（相对 L2 误差）。
import sys, numpy as np

def load(path):
    return np.fromfile(path, dtype=np.float32)

def rel_err(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    denom = np.sqrt(np.mean(b*b))
    if denom == 0: denom = 1.0
    return np.sqrt(np.mean((a-b)**2)) / denom

if __name__ == "__main__":
    mine_h = load(sys.argv[1])   # C++ hidden [H]
    ref_h  = load(sys.argv[2])   # ref _cpp_hidden.bin [H]
    mine_q = load(sys.argv[3])   # C++ q_roped [n_q*head_dim]
    ref_q  = load(sys.argv[4])   # ref _cpp_q_roped.bin [n_q*head_dim]
    print(f"[diff] hidden    mine={mine_h.shape} ref={ref_h.shape} rel_err={rel_err(mine_h,ref_h)*100:.4f}%")
    print(f"[diff] q_roped   mine={mine_q.shape} ref={ref_q.shape} rel_err={rel_err(mine_q,ref_q)*100:.4f}%")
    # 顶号对齐检查
    print(f"[diff] hidden    max|diff|={np.max(np.abs(mine_h.astype(np.float64)-ref_h.astype(np.float64))):.6f}")
    print(f"[diff] q_roped   max|diff|={np.max(np.abs(mine_q.astype(np.float64)-ref_q.astype(np.float64))):.6f}")
