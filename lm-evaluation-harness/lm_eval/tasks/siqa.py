from lm_eval.base import MultipleChoiceTask


class Siqa(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "social_i_qa"
    DATASET_NAME = "social_i_qa"

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
        query = f"Please choose the correct answer to the question: {doc['context']} {doc['question']}\n(A) {doc['answerA']} (B) {doc['answerB']} (C) {doc['answerC']}\nAnswer:"
        out_doc = {
            "query": query,
            "choices":  ["A", "B", "C"],
            "gold": ["1", "2", "3"].index(doc["label"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]