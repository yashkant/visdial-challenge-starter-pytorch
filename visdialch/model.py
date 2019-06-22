from torch import nn

from visdialch.decoders import Decoder
from visdialch.encoders import Encoder
from visdialch.utils.checkpointing import load_checkpoint


class EncoderDecoderModel(nn.Module):
    """Convenience wrapper module, wrapping Encoder and Decoder modules.

    Parameters
    ----------
    encoder: nn.Module
    decoder: nn.Module
    """

    def __init__(self, model_config, vocabulary):
        super().__init__()
        # Pass vocabulary to construct Embedding layer.
        self.encoder = Encoder(model_config, vocabulary)
        self.decoder = Decoder(model_config, vocabulary)

        print("Encoder: {}".format(model_config["encoder"]))
        print("Decoder: {}".format(model_config["decoder"]))

        # Share word embedding between encoder and decoder.
        decoder.word_embed = encoder.word_embed

    def forward(self, batch):
        encoder_output = self.encoder(batch)
        decoder_output = self.decoder(encoder_output, batch)
        return decoder_output

    def load_checkpoint(self, load_pthpath):
        model_state_dict, _ = load_checkpoint(load_pthpath)
        if isinstance(self, nn.DataParallel):
            self.module.load_state_dict(model_state_dict)
        else:
            self.load_state_dict(model_state_dict)
        print("Loaded model from {}".format(args.load_pthpath))

