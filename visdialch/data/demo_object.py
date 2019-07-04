import os
import torch

from typing import Any, Dict, Optional
from mosestokenizer import MosesDetokenizer
from nltk.tokenize import word_tokenize
from torch.nn.functional import normalize
from visdialch.data import VisDialDataset
from urllib.parse import urlparse


# TODO: Add docstrings, hints
def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


class DemoObject:

    def __init__(
            self,
            caption_model,
            enc_dec_model,
            vocabulary,
            config: Dict[str, Any],
            cuda_device,
            add_boundary_toks: bool = True,
    ):
        super().__init__()
        self.detect_caption_model = caption_model
        self.enc_dec_model = enc_dec_model
        self.vocabulary = vocabulary
        self.dataset_config = config["dataset"]
        self.caption_config = config["captioning"]
        self.cuda_device = cuda_device
        self.add_boundary_toks = add_boundary_toks

        # Initialize class variables
        self.image_features, self.image_caption_nl, self.image_caption = (
            None, None, None
        )
        self.questions, self.question_lengths = [], []
        self.answers, self.answer_lengths = [], []
        self.history, self.history_lengths = [], []
        self.num_rounds = 0

    # Call this method to retrive a dict object for inference. Pass the
    # natural language question asked by the user as arg.
    def _get_data(self, question: Optional[str] = None):
        data = {}
        data["img_feat"] = self.image_features

        # only pick the last entry as we process a single question at a time
        data["hist"] = self.history[-1].view(1, 1, -1).long()
        data["hist_len"] = torch.tensor([self.history_lengths[-1]]).long()

        # process the question and fill the inference dict object
        if question is not None:
            question = word_tokenize(question)
            question = self.vocabulary.to_indices(question)
            pad_question, question_length = VisDialDataset._pad_sequences(
                self.dataset_config,
                self.vocabulary,
                [question]
            )
            data["ques"] = pad_question.view(1, 1, -1).long()
            data["ques_len"] = torch.tensor(question_length).long()

        ans_in = torch.tensor([self.vocabulary.SOS_INDEX]).long()
        data["ans_in"] = ans_in.view(1, 1, -1)

        return data

    # Call this method as we have new dialogs (ques/ans pairs) in conversation.
    def update(
            self,
            question: Optional[str] = None,
            answer: Optional[str] = None,
    ):
        if question is not None:
            question = word_tokenize(question)
            question = self.vocabulary.to_indices(question)
            self.questions.append(question)
            self.question_lengths.append(len(question))

        if answer is not None:
            answer = word_tokenize(answer)
            answer = self.vocabulary.to_indices(answer)
            self.answers.append(answer)
            self.answer_lengths.append(len(answer))

        # history does not take in padded inputs! 
        self.history, self.history_lengths = VisDialDataset._get_history(
            self.dataset_config,
            self.vocabulary,
            self.image_caption,
            self.questions,
            self.answers,
            False
        )
        self.num_rounds += 1

    # Call this method to reset data, this is used internally by set_image()
    def _reset(self):
        self.image_features, self.image_caption_nl, self.image_caption = (
            None, None, None
        )
        self.questions, self.question_lengths = [], []
        self.answers, self.answer_lengths = [], []
        self.history, self.history_lengths = [], []
        self.num_rounds = 0

    # Download, extract features and build caption for the image
    def set_image(self, image_path):
        self._reset()
        if not os.path.isabs(image_path) and not validate_url(image_path):
            image_path = os.path.abspath(image_path)
        print(f"Loading image from : {image_path}")
        caption_tokens, image_features = self.detect_caption_model.predict(
            image_path,
            self.caption_config["detectron_model"]["feat_name"],
            True,
        )

        if self.dataset_config["img_norm"]:
            image_features = normalize(image_features, dim=0, p=2)

        self.image_caption_nl = \
        self.detect_caption_model.pythia_model.caption_processor(
            caption_tokens.tolist()[0]
        )["caption"]
        self.image_caption = self.vocabulary.to_indices(
            word_tokenize(self.image_caption_nl))
        self.image_features = image_features.unsqueeze(0)
        # build the initial history
        self.update()

    # Returns natural language caption
    def get_caption(self):
        if self.image_caption_nl is not None:
            return self.image_caption_nl
        else:
            raise TypeError("Image caption not found. Make sure set_image is "
                            "called prior to using this command.")

    # Respond to the user question
    def respond(self, user_question):
        batch = self._get_data(user_question)
        for key in batch:
            batch[key] = batch[key].to(self.cuda_device)

        with torch.no_grad():
            (eos_flag, max_len_flag), output = self.enc_dec_model(batch)
        output = [word_idx.item() for word_idx in output.reshape(-1)]
        answer = self.vocabulary.to_words(output)

        # Throw away the trailing '<EOS>' tokens
        if eos_flag:
            first_eos_idx = answer.index(self.vocabulary.EOS_TOKEN)
            answer = answer[:first_eos_idx]

        # MosesDetokenizer used to detokenize, it is separated from nltk.
        # Refer: https://pypi.org/project/mosestokenizer/
        with MosesDetokenizer('en') as detokenize:
            answer = detokenize(answer)
        return answer
