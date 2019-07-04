from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

# Build Detection Model
class DetectionModel:

    def __init__(self, caption_config, cuda_device):
        self.caption_config = caption_config
        self.cuda_device = cuda_device
        self.detection_model = self._build_detection_model()

    def _build_detection_model(self):

        cfg.merge_from_file(
            self.caption_config["detectron_model"]["config_yaml"])
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(
            self.caption_config["detectron_model"]["model_pth"],
            map_location=self.cuda_device)

        load_state_dict(model, checkpoint.pop("model"))

        model.to(self.cuda_device)
        model.eval()
        return model
