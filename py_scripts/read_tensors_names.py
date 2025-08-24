from safetensors import safe_open
import sys

def list_tensors(file_path: str):
    with safe_open(file_path, framework="numpy") as f:
        print(f"Tensors in {file_path}:")
        for name in f.keys():
            tensor = f.get_tensor(name)
            print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")

if __name__ == "__main__":
    list_tensors("../models/story/model.safetensors")
