import argparse

import torch
import yaml
from mosestokenizer import MosesDetokenizer
from torch import nn

from visdialch.data.demo_object import DemoObject
from visdialch.decoders import Decoder
from visdialch.encoders import Encoder
from visdialch.metrics import SparseGTMetrics, NDCG
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import load_checkpoint
from captioning.caption import PythiaCaptioning

parser = argparse.ArgumentParser(
    "Evaluate and/or generate EvalAI submission file."
)
parser.add_argument(
    "--config-yml",
    default="configs/lf_disc_faster_rcnn_x101.yml",
    help="Path to a config file listing reader, model and optimization "
         "parameters.",
)

parser.add_argument_group("Demo related arguments")
parser.add_argument(
    "--load-pthpath",
    default="checkpoints/checkpoint_xx.pth",
    help="Path to .pth file of pretrained checkpoint.",
)

# parser.add_argument(
#     "--load-imagepath",
#     default="checkpoints/checkpoint_xx.pth",
#     help="Path to .pth file of pretrained checkpoint.",
# )

parser.add_argument_group(
    "Arguments independent of experiment reproducibility"
)
parser.add_argument(
    "--gpu-ids",
    nargs="+",
    type=int,
    default=-1,
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
#   LOAD MODEL
# =============================================================================

demo_object = DemoObject(config["dataset"])

# Pass vocabulary to construct Embedding layer.
encoder = Encoder(config["model"], demo_object.vocabulary)
decoder = Decoder(config["model"], demo_object.vocabulary)

print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

# Share word embedding between encoder and decoder.
decoder.word_embed = encoder.word_embed

# Wrap encoder and decoder in a model. Don't use nn.DataParallel
# since batch_size is 1 for all runs.
model = EncoderDecoderModel(encoder, decoder).to(device)

model_state_dict, _ = load_checkpoint(args.load_pthpath)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(model_state_dict)
else:
    model.load_state_dict(model_state_dict)
print("Loaded model from {}".format(args.load_pthpath))

# =============================================================================
#   EVALUATION LOOP
# =============================================================================

model.eval()
break_loop = False

input_image_caption = input("Enter Caption: ").lower()
demo_object.update(caption=input_image_caption)
while not break_loop:
    user_question = input("Type Question: ").lower()
    batch = demo_object.get_data(user_question)

    for key in batch:
        batch[key] = batch[key].to(device)

    with torch.no_grad():
        (eos_flag, max_len_flag), output = model(batch)
    output = [word_idx.item() for word_idx in output.reshape(-1)]
    answer = demo_object.vocabulary.to_words(output)

    # Throw away the trailing '<EOS>' tokens
    if eos_flag:
        first_eos_idx = answer.index(demo_object.vocabulary.EOS_TOKEN)
        answer = answer[:first_eos_idx]
    
    # MosesDetokenizer used to detokenize, it is separated from nltk.
    # Refer: https://pypi.org/project/mosestokenizer/
    with MosesDetokenizer('en') as detokenize:
        answer = detokenize(answer)
    print(f"Answer: {answer}")
    demo_object.update(question=user_question, answer=answer)
