import torch
import time
import argparse

def keep_alive(device_id=0, matrix_size=4096, sleep_time=0.1):
    print(f"Starting GPU keep-alive on cuda:{device_id}...")
    print(f"Matrix size: {matrix_size}x{matrix_size}, Sleep interval: {sleep_time}s")
    print("Press Ctrl+C to stop.")
    
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cpu":
        print("Warning: CUDA not available, running on CPU instead.")
        
    # 创建两个随机矩阵
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    
    try:
        while True:
            # 执行矩阵乘法（消耗 GPU 算力）
            c = torch.matmul(a, b)
            # 同步，确保计算完成
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            # 短暂休眠，防止 GPU 100% 满载导致过热或影响其他轻量级操作
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size (larger = more VRAM & load)")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep time between ops (smaller = higher load)")
    args = parser.parse_args()
    
    keep_alive(args.device, args.size, args.sleep)
