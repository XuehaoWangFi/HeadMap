"""
WinoGrande: An Adversarial Winograd Schema Challenge at Scale
https://arxiv.org/pdf/1907.10641.pdf

WinoGrande is a collection of 44k problems, inspired by Winograd Schema Challenge
(Levesque, Davis, and Morgenstern 2011), but adjusted to improve the scale and
robustness against the dataset-specific bias. Formulated as a fill-in-a-blank
task with binary options, the goal is to choose the right option for a given
sentence which requires commonsense reasoning.

NOTE: This evaluation of Winogrande uses partial evaluation as described by
Trinh & Le in Simple Method for Commonsense Reasoning (2018).
See: https://arxiv.org/abs/1806.02847

Homepage: https://leaderboard.allenai.org/winogrande/submissions/public
"""
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@article{sakaguchi2019winogrande,
    title={WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
    author={Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
    journal={arXiv preprint arXiv:1907.10641},
    year={2019}
}
"""


class Winogrande(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "winogrande"
    DATASET_NAME = "winogrande_xl"

    answer_to_num = {"1": 0, "2": 1}

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        query = f"Please choose the correct answer to fill in the blank to complete the given sentence: {doc['sentence']}\n(A) {doc['option1']} (B) {doc['option2']}\nAnswer:"
        out_doc = {
            "query": query,
            "choices":  ["A", "B"],
            "gold": ["1", "2"].index(doc["answer"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]