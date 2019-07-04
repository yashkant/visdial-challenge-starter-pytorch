from .utils import get_detectron_features
from pythia.common.sample import Sample, SampleList
import gc
import torch

# Wrapper class over Detection and Captioning model that defines the
# combined the caption generation from raw image path
class DetectCaption:

    def __init__(self, detection_model, caption_model, cuda_device):
        self.detection_model = detection_model
        self.pythia_model = caption_model
        self.cuda_device = cuda_device

    def predict(self, url, feat_name, get_features=False):
        with torch.no_grad():
            detectron_features = get_detectron_features(
                [url],
                self.detection_model,
                False,
                feat_name,
                self.cuda_device
            )
            # returns a single-element list
            detectron_features = detectron_features[0]

            sample = Sample()
            sample.dataset_name = "coco"
            sample.dataset_type = "test"
            sample.image_feature_0 = detectron_features
            sample.answers = torch.zeros((5, 10), dtype=torch.long)

            sample_list = SampleList([sample])
            sample_list = sample_list.to(self.cuda_device)

            tokens = self.pythia_model(sample_list)["captions"]

        gc.collect()
        torch.cuda.empty_cache()

        if not get_features:
            return tokens
        else:
            return tokens, detectron_features
