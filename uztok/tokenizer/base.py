from abc import abstractmethod
from typing import List


class BaseTokenizer:

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError(">>> tokenizer::tokenize() isn\'t implemented.")