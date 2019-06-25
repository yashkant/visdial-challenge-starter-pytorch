import argparse
import glob
import os
import torch
import cv2  # must import before importing caffe2 due to bug in cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
import yaml
import sys
import numpy as np

sys.path.append("..")  # add parent to import modules
import os
# path to packages inside captioning are already available to interpreter
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import (
    build_detection_model
)
from maskrcnn_benchmark.utils.model_serialization import (
    load_state_dict
)
from captioning.utils import process_feature_extraction
from visdialch.data.dataset import RawImageDataset

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser(
    description="Extract bottom-up features from a model trained by Detectron"
)
parser.add_argument(
    "--image-root",
    nargs="+",
    help="Path to a directory containing COCO/VisDial images. Note that this "
         "directory must have images, and not sub-directories of splits. "
         "Each HDF file should contain features from a single split."
         "Multiple paths are supported to account for VisDial v1.0 train.",
)
parser.add_argument(
    "--config",
    help="Path to model config file used by Detectron (.yaml)",
    default="configs/lf_gen_faster_rcnn_x101_demo.yml",
)

parser.add_argument(
    "--save-path",
    help="Path to output file for saving bottom-up features (.h5)",
    default="data_img_mask_rcnn_x101.h5",
)
parser.add_argument(
    "--max-boxes",
    help="Maximum number of bounding box proposals per image",
    type=int,
    default=100
)
parser.add_argument(
    "--feat-name",
    help="The name of the layer to extract features from.",
    default="fc7",
)
parser.add_argument(
    "--feat-dims",
    help="Length of bottom-upfeature vectors.",
    type=int,
    default=2048,
)
parser.add_argument(
    "--split",
    choices=["train", "val", "test"],
    help="Which split is being processed.",
)
parser.add_argument(
    "--gpu-id",
    help="The GPU id to use (-1 for CPU execution)",
    type=int,
    default=0,
)
parser.add_argument(
    "--batch-size",
    help="Batch size for no. of images to be processed in one iteration",
    type=int,
    default=8,
)


def image_id_from_path(image_path):
    """Given a path to an image, return its id.

    Parameters
    ----------
    image_path : str
        Path to image, e.g.: coco_train2014/COCO_train2014/000000123456.jpg

    Returns
    -------
    int
        Corresponding image id (123456)
    """

    return int(image_path.split("/")[-1][-16:-4])


def main(args):
    """Extract bottom-up features from all images in a directory using
    a pre-trained Detectron model, and save them in HDF format.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    # load config
    # TODO: We just require captioning config only, so let's keep a local file
    # TODO: Also add batch_size and other extraction specific configs to file
    visdial_path = os.getcwd() + "/../"
    config = yaml.load(open(visdial_path + args.config))
    caption_config = config["captioning"]
    cfg.merge_from_file(
        visdial_path + caption_config["detectron_model"]["config_yaml"]
    )
    cfg.freeze()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    device = (
        torch.device("cuda", args.gpu_ids[0])
        if args.gpu_ids[0] >= 0
        else torch.device("cpu")
    )

    # TODO: pretty print config and use get_abspath, put config file in cwd
    # build mask-rcnn detection model
    detection_model = build_detection_model(cfg)
    detection_model.to(device)
    if -1 not in args.gpu_ids:
        detection_model = torch.nn.DataParallel(detection_model, args.gpu_ids)

    checkpoint = torch.load(
        visdial_path + caption_config["detectron_model"]["model_pth"],
        map_location=device)

    load_state_dict(detection_model, checkpoint.pop("model"))
    detection_model.eval()

    # create an output HDF to save extracted features
    save_h5 = h5py.File(args.save_path, "w")
    image_ids_h5d = save_h5.create_dataset(
        "image_ids", (len(image_paths),), dtype=int
    )

    boxes_h5d = save_h5.create_dataset(
        "boxes", (len(image_paths), args.max_boxes, 4),
    )
    features_h5d = save_h5.create_dataset(
        "features", (len(image_paths), args.max_boxes, args.feat_dims),
    )
    classes_h5d = save_h5.create_dataset(
        "classes", (len(image_paths), args.max_boxes,),
    )
    scores_h5d = save_h5.create_dataset(
        "scores", (len(image_paths), args.max_boxes,),
    )



    # # list of paths (example: "coco_train2014/COCO_train2014_000000123456.jpg")
    # image_paths = []
    # for image_root in args.image_root:
    #     image_paths.extend(
    #         [
    #             os.path.join(image_root, name)
    #             for name in glob.glob(os.path.join(image_root, "*.jpg"))
    #             if name not in {".", ".."}
    #         ]
    #     )

    raw_image_dataset = RawImageDataset(args.image_root, args.split)
    raw_image_dataloader = DataLoader(
        raw_image_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
    )

    for i, batch in enumerate(tqdm(raw_image_dataloader)):
        print(f"Batch no: {i}")
        batch_size = len(batch)

        # calculate idx_start and idx_end
        idx_start, idx_end = i * batch_size, (i + 1) * batch_size

        # shape: ( batch_size, dict )
        im_scales = []
        for key in batch:
            batch[key] = batch[key].to(device)
            im_scales.append(batch[key]["image_scale"])

        with torch.no_grad():
            output = detection_model(batch)

        boxes, features, classes, scores = process_feature_extraction(
            output,
            im_scales,
            get_boxes=get_boxes,
            feat_name=feat_name,
            conf_thresh=0.2
        )
        boxes_h5d[idx_start:idx_end] = np.array(
            [item.cpu().numpy() for item in boxes])
        features_h5d[idx_start:idx_end] = np.array(
            [item.cpu().numpy() for item in features])
        classes_h5d[idx_start:idx_end] = np.array(
            [item.cpu().numpy() for item in classes])
        scores_h5d[idx_start:idx_end] = np.array(
            [item.cpu().numpy() for item in classes])





    # batch_size = args.batch_size
    # mini_batches = len(image_paths) // batch_size
    # for iter in tqdm(range(mini_batches + 1), desc="Processing Batches"):
    #     print(f"Iteration: {iter}")
    #     idx_start, idx_end = iter * batch_size, (iter + 1) * batch_size
    #     batch_image_paths = image_paths[idx_start:idx_end]
    #     boxes, features, classes, scores = get_detectron_features(
    #         batch_image_paths,
    #         detection_model,
    #         True,
    #         args.feat_name,
    #         device,
    #         batch_mode=True
    #     )
    #     boxes_h5d[idx_start:idx_end] = np.array(
    #         [item.cpu().numpy() for item in boxes])
    #     features_h5d[idx_start:idx_end] = np.array(
    #         [item.cpu().numpy() for item in features])
    #     classes_h5d[idx_start:idx_end] = np.array(
    #         [item.cpu().numpy() for item in classes])
    #     scores_h5d[idx_start:idx_end] = np.array(
    #         [item.cpu().numpy() for item in classes])

    # set current split name in attributrs of file, for tractability
    save_h5.attrs["split"] = args.split
    save_h5.close()


if __name__ == "__main__":
    # set higher log level to prevent terminal spam
    args = parser.parse_args()
    main(args)
