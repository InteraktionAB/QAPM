"""
pegasus

This module contain various Pegasus implementations.
"""

import abstract
import functools
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

class HFPegasus(abstract.Paraphrase):

    """
    HFPegasus

    Implementation of Pegasus.
    """

    def __init__(self,):

        """
        HFPegasus

        Parameters

        Returns
        """

        model_name = 'tuner007/pegasus_paraphrase'
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.torch_device)

    def infer(self, input_text, num_return_sequences, num_beams, max_length=60, temperature=1.5):
        batch = self.tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(self.torch_device)
        translated = self.model.generate(**batch,max_length=max_length,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=temperature)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def run(self, texts: list,):

        """
        run

        Parameters

        Returns
        """

        infer_ = functools.partial(self.infer, num_return_sequences=10, num_beams=10)
        return list(map(infer_, texts))