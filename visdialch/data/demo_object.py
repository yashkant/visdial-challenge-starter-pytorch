from typing import Any, Dict, Optional

import torch
from nltk.tokenize import word_tokenize
from torch.nn.functional import normalize

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

        # Store image features of the selected image in our object
        image_id = self.hdf_reader.keys()[-14]
        print(f"Image id: {int(image_id)}")
        image_features = torch.tensor(self.hdf_reader[image_id])
        
        if self.config["img_norm"]:
            image_features = normalize(image_features, dim=0, p=2)
        
        self.image_features = image_features.unsqueeze(0)
        
        # Make the call for the generating caption here.
        image_caption = "a group with drinks posing for a picture"

        # Store the caption in natural language
        self.image_caption_nl = image_caption
        image_caption = word_tokenize(image_caption)
        self.image_caption = self.vocabulary.to_indices(image_caption)

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
        data["hist"] = self.history[-1].view(1,1,-1).long()
        data["hist_len"] = torch.tensor([self.history_lengths[-1]]).long()

        if question is not None:
            # Create field for current question, I think we need to place
            # questions one by one in data["ques"] field and move the
            # older ones to history (done by update).
            question = word_tokenize(question)
            question = self.vocabulary.to_indices(question)
            pad_question, question_length = VisDialDataset._pad_sequences(
                self.config,
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
            caption: Optional[str] = None
    ):
        if caption is not None:
            caption = word_tokenize(caption)
            self.image_caption = self.vocabulary.to_indices(caption)

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
            self.config,
            self.vocabulary,
            self.image_caption,
            self.questions,
            self.answers,
            False
        )
        self.num_rounds += 1

    # Call this method to reset data
    def reset(self):
        pass

    # Returns natural language caption
    def get_caption(self):
        return self.image_caption_nl
