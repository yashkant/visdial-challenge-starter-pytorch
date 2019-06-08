import torch
from torch import nn


class GenerativeDecoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config
        self.vocabulary = vocabulary

        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.answer_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )

        # Handle forward methods by mode in config.
        self.mode = None
        if "mode" in config:
            self.mode = config["mode"]

        self.lstm_to_words = nn.Linear(
            self.config["lstm_hidden_size"], len(vocabulary)
        )

        self.dropout = nn.Dropout(p=config["dropout"])
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_output, batch):
        """Given `encoder_output`, learn to autoregressively predict
        ground-truth answer word-by-word during training.

        During evaluation, assign log-likelihood scores to all answer options.

        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """

        if self.training:

            ans_in = batch["ans_in"]
            batch_size, num_rounds, max_sequence_length = ans_in.size()

            ans_in = ans_in.view(batch_size * num_rounds, max_sequence_length)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         word_embedding_size)
            ans_in_embed = self.word_embed(ans_in)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, batch_size * num_rounds,
            #         lstm_hidden_size)
            init_hidden = encoder_output.view(1, batch_size * num_rounds, -1)
            init_hidden = init_hidden.repeat(
                self.config["lstm_num_layers"], 1, 1
            )
            init_cell = torch.zeros_like(init_hidden)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         lstm_hidden_size)
            ans_out, (hidden, cell) = self.answer_rnn(
                ans_in_embed, (init_hidden, init_cell)
            )
            ans_out = self.dropout(ans_out)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         vocabulary_size)
            ans_word_scores = self.lstm_to_words(ans_out)
            return ans_word_scores

        elif self.mode is not None and self.mode == "demo":

            # batch_size, num_rounds are 1 throughout the demo loop
            batch_size, num_rounds = 1, 1
            ans_in = batch["ans_in"]

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, 1 * 1, lstm_hidden_size)
            hidden = encoder_output.view(1, batch_size * num_rounds, -1)
            hidden = hidden.repeat(
                self.config["lstm_num_layers"], 1, 1
            )
            cell = torch.zeros_like(hidden)

            # stop when any of the flags below is raised
            end_token_flag = False
            max_seq_len_flag = False
            answer_indices = []

            while end_token_flag is False and max_seq_len_flag is False:

                # shape: (1*1, sequence_length)
                ans_in = ans_in.view(batch_size * num_rounds, -1)

                # shape: (1*1, sequence_length, word_embedding_size)
                ans_in_embed = self.word_embed(ans_in)

                # shape: (1*1, sequence_length, lstm_hidden_size)
                # new states are updated in (hidden, cell)
                ans_out, (hidden, cell) = self.answer_rnn(
                    ans_in_embed, (hidden, cell)
                )
                
                # get the ans-idx from logits
                # ans_out = self.dropout(ans_out)
                ans_scores = self.logsoftmax(self.lstm_to_words(ans_out))
                ans_scores = torch.exp(ans_scores)

                # TODO: remove <PAD>, <S> (for t > 0) and </S> (for t == 0)
                _, ans_index = ans_scores.view(-1).max(0)
                answer_indices.append(ans_index)
                ans_in = ans_index

                # check flag conditions and raise
                if ans_index.item() == self.vocabulary.EOS_INDEX:
                    end_token_flag = True
                if len(answer_indices) >= 20:
                    max_seq_len_flag = True

            return (end_token_flag, max_seq_len_flag), answer_indices

        else:

            ans_in = batch["opt_in"]
            batch_size, num_rounds, num_options, max_sequence_length = (
                ans_in.size()
            )

            ans_in = ans_in.view(
                batch_size * num_rounds * num_options, max_sequence_length
            )

            # shape: (batch_size * num_rounds * num_options, max_sequence_length
            #         word_embedding_size)
            ans_in_embed = self.word_embed(ans_in)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, batch_size * num_rounds * num_options,
            #         lstm_hidden_size)
            init_hidden = encoder_output.view(batch_size, num_rounds, 1, -1)
            init_hidden = init_hidden.repeat(1, 1, num_options, 1)
            init_hidden = init_hidden.view(
                1, batch_size * num_rounds * num_options, -1
            )
            init_hidden = init_hidden.repeat(
                self.config["lstm_num_layers"], 1, 1
            )
            init_cell = torch.zeros_like(init_hidden)

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length, lstm_hidden_size)
            ans_out, (hidden, cell) = self.answer_rnn(
                ans_in_embed, (init_hidden, init_cell)
            )

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length, vocabulary_size)
            ans_word_scores = self.logsoftmax(self.lstm_to_words(ans_out))

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length)
            target_ans_out = batch["opt_out"].view(
                batch_size * num_rounds * num_options, -1
            )

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length)
            ans_word_scores = torch.gather(
                ans_word_scores, -1, target_ans_out.unsqueeze(-1)
            ).squeeze()
            ans_word_scores = (
                ans_word_scores * (target_ans_out > 0).float().cuda()
            )  # ugly

            ans_scores = torch.sum(ans_word_scores, -1)
            ans_scores = ans_scores.view(batch_size, num_rounds, num_options)

            return ans_scores
