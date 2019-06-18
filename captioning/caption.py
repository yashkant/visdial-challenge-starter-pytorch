import gc
import torch
import torch.nn.functional as F
import yaml
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from pythia.common.registry import registry
from pythia.common.sample import Sample, SampleList
from pythia.models.butd import BUTD
from pythia.tasks.processors import VocabProcessor, CaptionProcessor
from pythia.utils.configuration import ConfigNode
from utils import get_actual_image, image_transform, process_feature_extraction

class PythiaCaptioning:
    TARGET_IMAGE_SIZE = [448, 448]
    CHANNEL_MEAN = [0.485, 0.456, 0.406]
    CHANNEL_STD = [0.229, 0.224, 0.225]

    def __init__(self, caption_config):
        self.caption_config = caption_config
        self._init_processors()
        self.pythia_model = self._build_pythia_model()
        self.detection_model = self._build_detection_model()

    def _init_processors(self):
        with open(self.caption_config["butd_model"]["config_yaml"]) as f:
            config = yaml.load(f)

        config = ConfigNode(config)
        # Remove warning
        config.training_parameters.evalai_inference = True
        registry.register("config", config)

        self.config = config

        captioning_config = config.task_attributes.captioning.dataset_attributes.coco
        text_processor_config = captioning_config.processors.text_processor
        caption_processor_config = captioning_config.processors.caption_processor
        vocab_file_path = self.caption_config[
            "text_caption_processor_vocab_txt"]
        text_processor_config.params.vocab.vocab_file = vocab_file_path
        caption_processor_config.params.vocab.vocab_file = vocab_file_path
        self.text_processor = VocabProcessor(text_processor_config.params)
        self.caption_processor = CaptionProcessor(
            caption_processor_config.params)

        registry.register("coco_text_processor", self.text_processor)
        registry.register("coco_caption_processor", self.caption_processor)

    def _build_pythia_model(self):
        state_dict = torch.load(self.caption_config["butd_model"]["model_pth"])
        model_config = self.config.model_attributes.butd
        model_config.model_data_dir = self.caption_config["model_data_dir"]
        model = BUTD(model_config)
        model.build()
        model.init_losses_and_metrics()

        if list(state_dict.keys())[0].startswith('module') and \
                not hasattr(model, 'module'):
            state_dict = self._multi_gpu_state_to_single(state_dict)

        model.load_state_dict(state_dict)
        model.to("cuda")
        model.eval()

        return model

    def _multi_gpu_state_to_single(self, state_dict):
        new_sd = {}
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                raise TypeError("Not a multiple GPU state of dict")
            k1 = k[7:]
            new_sd[k1] = v
        return new_sd

    def predict(self, url):
        with torch.no_grad():
            detectron_features = self.get_detectron_features(url)

            sample = Sample()
            sample.dataset_name = "coco"
            sample.dataset_type = "test"
            sample.image_feature_0 = detectron_features
            sample.answers = torch.zeros((5, 10), dtype=torch.long)

            sample_list = SampleList([sample])
            sample_list = sample_list.to("cuda")

            tokens = self.pythia_model(sample_list)["captions"]

        gc.collect()
        torch.cuda.empty_cache()

        return tokens

    def _build_detection_model(self):

        cfg.merge_from_file(
            self.caption_config["detectron_model"]["config_yaml"])
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(
            self.caption_config["detectron_model"]["model_pth"],
            map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def masked_unk_softmax(self, x, dim, mask_idx):
        x1 = F.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def get_detectron_features(self, image_path):
        im, im_scale = image_transform(image_path)
        img_tensor, im_scales = [im], [im_scale]
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to('cuda')
        with torch.no_grad():
            output = self.detection_model(current_img_list)
        feat_list = process_feature_extraction(output, im_scales,
                                                     'fc6', 0.2)
        return feat_list[0]
