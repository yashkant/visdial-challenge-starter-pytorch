import argparse
import os

import torch
import yaml

from captioning import PythiaCaptioning
from visdialch.data import Vocabulary
from visdialch.data.demo_manager import DemoSessionManager
from visdialch.model import EncoderDecoderModel

parser = argparse.ArgumentParser(
    "Run Visual-Dialog Demo"
)
parser.add_argument(
    "--config-yml",
    default="configs/lf_gen_faster_rcnn_x101_demo.yml",
    help="Path to a config file listing reader, visual dialog and captioning "
         "model parameters.",
)

parser.add_argument_group("Demo related arguments")

parser.add_argument(
    "--image-path",
    default="/nethome/ykant3/tmp/COCO_test2014_000000355148.jpg",
    help="Path to .pth file of pretrained checkpoint.",
)

parser.add_argument_group(
    "Arguments independent of experiment reproducibility"
)
parser.add_argument(
    "--gpu-ids",
    nargs="+",
    type=int,
    default=0,
    help="List of ids of GPUs to use.",
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=4,
    help="Number of CPU workers for reading data.",
)
parser.add_argument(
    "--overfit",
    action="store_true",
    help="Overfit model on 5 examples, meant for debugging.",
)
parser.add_argument(
    "--in-memory",
    action="store_true",
    help="Load the whole dataset and pre-extracted image features in memory. "
         "Use only in presence of large RAM, atleast few tens of GBs.",
)

# For reproducibility.
# Refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =============================================================================
#   INPUT ARGUMENTS AND CONFIG
# =============================================================================

args = parser.parse_args()
# get abs path
if not os.path.isabs(args.config_yml):
    args.config_yml = os.path.abspath(args.config_yml)

# keys: {"dataset", "model", "solver"}
config = yaml.load(open(args.config_yml))

if isinstance(args.gpu_ids, int):
    args.gpu_ids = [args.gpu_ids]
device = (
    torch.device("cuda", args.gpu_ids[0])
    if args.gpu_ids[0] >= 0
    else torch.device("cpu")
)

# Print config and args.
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))

# =============================================================================
#   BUILD VOCABULARY | LOAD MODELS: ENC-DEC, CAPTIONING
# =============================================================================
dataset_config = config["dataset"]
model_config = config["model"]
captioning_config = config["captioning"]

vocabulary = Vocabulary(
    dataset_config["word_counts_json"],
    min_count=dataset_config["vocab_min_count"]
)

# Build Encoder-Decoder model and load its checkpoint
enc_dec_model = EncoderDecoderModel(model_config, vocabulary).to(device)
enc_dec_model.load_checkpoint(model_config["model_pth"])

# Build the captioning model and load its checkpoint
# Path to the checkpoint is picked from captioning_config
caption_model = PythiaCaptioning(captioning_config, device)

# Pass the Captioning and Encoder-Decoder models and initialize DemoObject
demo_object = DemoSessionManager(
    caption_model,
    enc_dec_model,
    vocabulary,
    config,
    device
)

# =============================================================================
#   DEMO LOOP
# =============================================================================

# Switch dropout, batchnorm etc to the correct mode.
enc_dec_model.eval()

# Extract image features and build caption
demo_object.set_image(args.image_path)
print(f"Caption: {demo_object.get_caption()}")

while True:

    # Input question, respond to it and update history
    user_question = input("Type Question: ").lower()
    answer = demo_object.respond(user_question)
    print(f"Answer: {answer}")

    while True:
        user_input = input("Change Image? [(y)es/(n)o]: ").lower()
        if user_input == 'y' or user_input == 'yes':
            print("-"*50)
            user_image = input("Enter New Image Path: ")
            demo_object.set_image(user_image)
            print(f"Caption: {demo_object.get_caption()}")

        elif user_input == 'n' or user_input == 'no':
            break

