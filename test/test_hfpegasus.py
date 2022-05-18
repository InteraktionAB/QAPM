"""
test_hfpegasus

This module contains the test for HFPegasus.
"""

import pytest
import qapm

texts = ["dummy_text"]

@pytest.mark.parametrize("texts", texts)
def test_hfpegasus(texts):

    """
    test_hfpegasus

    This function will test the huggingface
    pegasus model.
    """

    pegasus = qapm.HFPegasus()
    results = pegasus.run(["dummy text"])

    assert len(results) == len(texts)