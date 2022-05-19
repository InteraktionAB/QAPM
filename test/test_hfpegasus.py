"""
test_hfpegasus

This module contains the test for HFPegasus.
"""

import pytest
import qapm

texts = ["The ultimate test of your knowledge is your capacity to convey it to another."]
pegasus = qapm.HFPegasus()

@pytest.mark.parametrize("pegasus_, texts_", [(pegasus, texts)])
def test_hfpegasus(pegasus_, texts_):

    """
    test_hfpegasus

    This function will test the huggingface
    pegasus model.
    """

    results = pegasus_.run(texts_)

    assert len(results) == len(texts_)
    assert results != texts_