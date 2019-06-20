from typing import Any, Dict, Optional

import torch
from nltk.tokenize import word_tokenize
from torch.nn.functional import normalize
import os
from visdialch.data import VisDialDataset
from visdialch.data.readers import (
    ImageFeaturesHdfReader,
)
from visdialch.data.vocabulary import Vocabulary
from captioning.caption import PythiaCaptioning
from mosestokenizer import MosesDetokenizer


class DemoObject:

    def __init__(
            self,
            config: Dict[str, Any],
            in_memory: bool = False,
            add_boundary_toks: bool = True,
    ):
        super().__init__()
        self.dataset_config = config["dataset"]
        self.captioning_config = config["captioning"]
        self.add_boundary_toks = add_boundary_toks

        self.vocabulary = Vocabulary(
            config["word_counts_json"], min_count=config["vocab_min_count"]
        )

        # build captioning and feature extraction model
        self.caption_model = PythiaCaptioning(self.captioning_config)
        self.image_features, self.image_caption_nl, self.image_caption = (
            None, None, None
        )

        # TODO: Think about this norm thingy below! Do I need to use this on
        # our current features! I think yes, because it will be used
        # when we train the visdial-model on new features.

        # if self.config["img_norm"]:
        #     image_features = normalize(image_features, dim=0, p=2)
        #

        # self.image_features = image_features.unsqueeze(0)

        self.questions, self.question_lengths = [], []
        self.answers, self.answer_lengths = [], []
        self.history, self.history_lengths = [], []
        self.num_rounds = 0
        self.update()

    # Call this method to retrive an object for inference, pass the
    # natural language question asked by the user.
    def get_data(self, question: Optional[str] = None):
        data = {}
        data["img_feat"] = self.image_features

        # only pick the last entry as we process a single question at a time
        data["hist"] = self.history[-1].view(1, 1, -1).long()
        data["hist_len"] = torch.tensor([self.history_lengths[-1]]).long()

        if question is not None:
            # Create field for current question, I think we need to place
            # questions one by one in data["ques"] field and move the
            # older ones to history (done by update).
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

    # Call this method as we have new dialogs in conversation.
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

    # Call this method to reset data
    def reset(self):
        # think about this
        pass

    # Extracts features and build caption for the image
    def set_image(self, image_path):
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)
        print("Path:", image_path)
        caption_tokens, image_features = self.caption_model.predict(
            image_path,
            True
        )
        if self.dataset_config["img_norm"]:
            image_features = normalize(image_features, dim=0, p=2)

        # MosesDetokenizer used to detokenize, it is separated from nltk.
        # Refer: https://pypi.org/project/mosestokenizer/
        with MosesDetokenizer('en') as detokenize:
            self.image_caption_nl = detokenize(caption_tokens)
        self.image_features = image_features
        self.image_caption = self.vocabulary.to_indices(caption_tokens)

    # Returns natural language caption
    def get_caption(self):
        if self.image_caption_nl is not None:
            return self.image_caption_nl
        else:
            # TODO: raise error here
            return None
