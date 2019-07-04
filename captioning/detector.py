import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model as _build_detection_model
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

# Build Detection Model

def build_detection_model(caption_config, cuda_device):
    cfg.merge_from_file(
        caption_config["detectron_model"]["config_yaml"])
    cfg.freeze()
    model = _build_detection_model(cfg)
    checkpoint = torch.load(
        caption_config["detectron_model"]["model_pth"],
        map_location=cuda_device)
    load_state_dict(model, checkpoint.pop("model"))
    model.to(cuda_device)
    model.eval()
    return model