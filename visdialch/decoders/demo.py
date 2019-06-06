from visdialch.decoders import GenerativeDecoder


class VisdialDemoDecoder(GenerativeDecoder):

    def __init__(self, config, vocabulary):
        super().__init__(config, vocabulary)

    def forward(self, encoder_output, batch):



        return super().forward(encoder_output, batch)
