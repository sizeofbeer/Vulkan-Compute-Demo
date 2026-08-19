import struct

def walk(path, files_desc):
    data = open(path,'rb').read()
    n = len(data)
    print(f"file {path} size={n}")
    off = 0
    for (lname, counts) in files_desc:
        for ci,c in enumerate(counts):
            if off+4 > n:
                print(f"  OVERFLOW at {lname} blob{ci} off={off}"); return
            tag = struct.unpack_from('<I', data, off)[0]
            if tag == 0x01306B47:
                padded = (c*2 + 3) & ~3
                adv = 4 + padded
                fmt='fp16'
                extra=""
            else:
                adv = c*4
                fmt='fp32'
                val = struct.unpack_from('<f', data, off)[0]
                extra=f"first={val:.5g}"
            print(f"  {lname} b{ci} count={c} off={off} tag=0x{tag:08X} {fmt} adv={adv} {extra}")
            off += adv
    print(f"  -> end off={off} filesize={n} diff={n-off}")

print("=== A: NO beta ===")
walk("decoder_00_pre.ncnn.bin", [("rmsn_3",[1024]),("gemm_0",[2048*1024]),("gemm_1",[1024*1024]),("gemm_2",[1024*1024])])
print("=== B: WITH beta ===")
walk("decoder_00_pre.ncnn.bin", [("rmsn_g",[1024]),("rmsn_beta",[1024]),("gemm_0",[2048*1024]),("gemm_1",[1024*1024]),("gemm_2",[1024*1024])])
