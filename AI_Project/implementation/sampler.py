"""Base sampler class for LLM sampling."""

from typing import Collection


class LLM:
    """Base class for language model sampling."""

    def __init__(self, samples_per_prompt: int):
        self._samples_per_prompt = samples_per_prompt

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        raise NotImplementedError
