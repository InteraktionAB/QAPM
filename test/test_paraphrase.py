"""
test_paraphrase
----------

This module contains the test for the abstract class
Paraphrase.
"""

import abc
import abstract

def test_paraphrase():
    
    """
    test_paraphrase

    Test if paraphrase is abstract or not.

    Parameters

    Returns
    """

    assert issubclass(abstract.Paraphrase, abc.ABC)