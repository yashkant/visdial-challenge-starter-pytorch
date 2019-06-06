# I think a single method that takes in image_path, caption and returns
# image features, history and empty questions place-holder works.

from typing import Any, Dict, List

import torch
from nltk.tokenize import word_tokenize

from visdialch.data import VisDialDataset
from visdialch.data.readers import (
    ImageFeaturesHdfReader,
)
from visdialch.data.vocabulary import Vocabulary


class DemoObject:

    def __init__(
            self,
            config: Dict[str, Any],
            in_memory: bool = False,
            add_boundary_toks: bool = True,
    ):
        super().__init__()
        self.config = config
        self.add_boundary_toks = add_boundary_toks

        self.vocabulary = Vocabulary(
            config["word_counts_json"], min_count=config["vocab_min_count"]
        )

        # Extract the image-features from the image-path here
        image_features_hdfpath = config["image_features_train_h5"]
        self.hdf_reader = ImageFeaturesHdfReader(
            image_features_hdfpath, in_memory
        )
        self.image_ids = list(self.dialogs_reader.dialogs.keys())

        # Store image features of the selected image in our object
        self.image_id = self.image_ids[-3]
        self.image_features = self.hdf_reader[self.selected_image_id]

        # Make the call for the generating caption here.
        image_caption = "There is a cow in a green" \
                        " field eating grass."
        image_caption = word_tokenize(image_caption)
        self.image_caption = self.vocabulary.to_indices(image_caption)

        self.questions, self.question_lengths = [], []
        self.answers, self.answer_lengths = [], []
        self.history, self.history_lengths = [], []
        self.num_rounds = 0
        self.update()

    # Call this method to retrive an object for inference.
    def get_data(self):
        data = {}
        data["img_feat"] = self.image_features
        data["ques"] = self.questions.long()
        data["hist"] = self.history.long()
        data["ques_len"] = torch.tensor(self.question_lengths).long()
        data["hist_len"] = torch.tensor(self.history_lengths).long()
        data["ans_len"] = torch.tensor(self.answer_lengths).long()
        data["num_rounds"] = torch.tensor(self.num_rounds).long()

        return data

    # Call this method as we have new dialogs in conversation.
    def update(self, question: str = None, answer: str = None):

        if question is not None:
            question = word_tokenize(question)
            question = self.vocabulary.to_indices(question)
            pad_question, question_length = VisDialDataset._pad_sequences(
                self.config,
                self.vocabulary,
                [question]
            )
            self.questions += pad_question
            self.question_lengths += question_length

        if answer is not None:
            answer = word_tokenize(answer)
            answer = self.vocabulary.to_indices(answer)
            pad_answer, answer_length = VisDialDataset._pad_sequences(
                self.config,
                self.vocabulary,
                [answer]
            )
            self.answers += pad_answer
            self.answer_lengths += answer_length

        self.history, self.history_lengths = VisDialDataset._get_history(
            self.image_caption,
            self.questions,
            self.answers
        )

        self.num_rounds += 1

    # Call this method to reset data
    def reset(self):
        pass
