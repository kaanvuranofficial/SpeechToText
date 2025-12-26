import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

class ActDropNormCNN1D(nn.Module):
    def __init__(self, n_feats, dropout, keep_shape=False):
        super(ActDropNormCNN1D, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(F.gelu(self.norm(x)))
        if self.keep_shape: return x.transpose(1, 2)
        else: return x

class SpeechRecognition(nn.Module):
    hyper_parameters = {
        "num_classes": 29, "n_feats": 81, "dropout": 0.1,
        "hidden_size": 1024, "num_layers": 1
    }
    def __init__(self, hidden_size=1024, num_classes=29, n_feats=81, num_layers=1, dropout=0.1):
        super(SpeechRecognition, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 10, 2, padding=10//2),
            ActDropNormCNN1D(n_feats, dropout),
        )
        self.dense = nn.Sequential(
            nn.Linear(n_feats, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, hidden=None):
        x = x.squeeze(1)
        x = self.cnn(x)
        x = self.dense(x)
        x = x.transpose(0, 1)
        out, _ = self.lstm(x, hidden)
        x = self.dropout2(F.gelu(self.layer_norm2(out)))
        return self.final_fc(x)

def apply_dynamic_quantization(model_path, output_path):
    print(f"[INFO] Loading model structure...")
    
    model = SpeechRecognition(**SpeechRecognition.hyper_parameters)
    
    if os.path.exists(model_path):
        print(f"[INFO] Loading weights from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"[WARNING] Could not load weights: {e}")
            print("[INFO] Proceeding with initialized weights for structure verification.")
    else:
        print(f"[WARNING] '{model_path}' not found. Initializing random weights for structure verification.")

    model.eval()

    print("[INFO] Applying Dynamic Quantization (Float32 -> Int8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
    )

    torch.save(quantized_model, output_path)
    print(f"[SUCCESS] Optimized model saved to: {output_path}")
    return quantized_model

def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

if __name__ == "__main__":
    input_model = "speech_recognition_model.ckpt"
    output_model = "optimized_model.pt"

    if not os.path.exists(input_model):
        print("[SYSTEM] Targeted model not found. Generating a dummy model for system test...")
        dummy_model = SpeechRecognition(**SpeechRecognition.hyper_parameters)
        torch.save(dummy_model.state_dict(), "temp_base_model.pt")
        input_model = "temp_base_model.pt"

    apply_dynamic_quantization(input_model, output_model)

    size_orig = get_file_size_mb(input_model)
    size_opt = get_file_size_mb(output_model)
    reduction = (1 - (size_opt / size_orig)) * 100

    print("-" * 50)
    print(f"OPTIMIZATION REPORT")
    print("-" * 50)
    print(f"Original Model Size : {size_orig:.2f} MB")
    print(f"Quantized Model Size: {size_opt:.2f} MB")
    print(f"Size Reduction      : {reduction:.2f}%")
    print("-" * 50)
    
    if input_model == "temp_base_model.pt":
        os.remove("temp_base_model.pt")
        print("[SYSTEM] Temporary test files cleaned up.")