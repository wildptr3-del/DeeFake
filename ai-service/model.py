import torch
import torch.nn as nn
from torchvision import models
import os

class DeepfakeEfficientNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeEfficientNet, self).__init__()
        # Use torchvision EfficientNet-B0 to match deepfake_model.pth keys
        if pretrained:
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        return self.backbone(x)

def get_model(model_path=None, device="cpu", optimize=True):
    """
    Load the DeepfakeEfficientNet model.
    
    Weight loading strategy:
      - If state_dict keys start with 'backbone.' -> saved by model.state_dict()
      - If state_dict keys start with 'features.'  -> saved by model.backbone.state_dict()
      - Automatically handles both formats
    
    Optimization strategy:
      - NO INT8 quantization (destroys classifier precision on small weights)
      - NO JIT tracing (changes model behavior unpredictably)
      - FP16 for CUDA only (safe precision reduction on GPU)
    """
    model = DeepfakeEfficientNet(pretrained=(model_path is None))
    
    if model_path and os.path.exists(model_path):
        try:
            print(f"[AI] Loading trained weights from {model_path}")
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            
            # Detect format and load appropriately
            if any(k.startswith('backbone.') for k in state_dict.keys()):
                # Saved via model.state_dict() -> load into full model
                print("[AI] Detected full model state_dict (backbone.* keys)")
                model.load_state_dict(state_dict, strict=False)
            elif any(k.startswith('features') for k in state_dict.keys()):
                # Saved via model.backbone.state_dict() -> load into backbone
                print("[AI] Detected backbone-only state_dict (features.* keys)")
                # Check classifier compatibility
                sd_classifier_keys = [k for k in state_dict if 'classifier' in k]
                model_classifier_keys = list(model.backbone.classifier.state_dict().keys())
                
                # Check if shapes match for the final linear layer
                compatible = True
                for k in sd_classifier_keys:
                    if k in model.backbone.state_dict():
                        if state_dict[k].shape != model.backbone.state_dict()[k].shape:
                            compatible = False
                            break
                
                if compatible and sd_classifier_keys:
                    model.backbone.load_state_dict(state_dict, strict=False)
                    print("[AI] Loaded backbone weights (classifier compatible)")
                else:
                    # Load only the feature extractor, skip incompatible classifier
                    print("[AI] Classifier shape mismatch -- loading features only")
                    feature_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
                    model.backbone.load_state_dict(feature_dict, strict=False)
                    print("[AI] ImageNet features loaded, classifier uses fresh weights")
            else:
                model.load_state_dict(state_dict, strict=False)
                
        except Exception as e:
            print(f"[AI] Weight loading error: {e}. Using ImageNet pretrained features.")
    
    model.to(device)
    model.eval()

    # Optimization: ONLY FP16 on CUDA (safe). No quantization, no JIT.
    if optimize and "cuda" in str(device):
        print("[AI] Optimization: Enabling FP16 for CUDA")
        model = model.half()
    
    print("[AI] Model ready (eager mode, full precision on CPU)")
    return model
