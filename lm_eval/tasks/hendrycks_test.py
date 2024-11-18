"""
Measuring Massive Multitask Language Understanding
https://arxiv.org/pdf/2009.03300.pdf

The Hendryck's Test is a benchmark that measured a text model’s multitask accuracy.
The test covers 57 tasks including elementary mathematics, US history, computer
science, law, and more. To attain high accuracy on this test, models must possess
extensive world knowledge and problem solving ability. By comprehensively evaluating
the breadth and depth of a model’s academic and professional understanding,
Hendryck's Test can be used to analyze models across many tasks and to identify
important shortcomings.

Homepage: https://github.com/hendrycks/test
"""
import os
import datasets
import csv
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@article{hendryckstest2021,
    title={Measuring Massive Multitask Language Understanding},
    author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021}
}
"""


SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {hendrycksTest-abstract_algebra: Task, hendrycksTest-anatomy: Task}
    """
    return {f"hendrycksTest-{sub}": create_task(sub) for sub in SUBJECTS}


def create_task(subject):
    class HendrycksTest(GeneralHendrycksTest):
        def __init__(self):
            super().__init__(subject)

    return HendrycksTest


class GeneralHendrycksTest(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = None
    DATASET_NAME = None
    URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
    
    def __init__(self, subject):
        self.DATASET_NAME = subject
        self._fewshot_docs = None
        self.download()  # Custom download instead of using parent's

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        """Custom download implementation"""
        try:
            print(f"Downloading dataset for {self.DATASET_NAME}")
            # Create cache directory if it doesn't exist
            cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/datasets")
            os.makedirs(cache_dir, exist_ok=True)
            
            print(f"Using cache directory: {cache_dir}")
            # Use datasets.download_manager instead
            dl_manager = datasets.DownloadManager(
                dataset_name="hendrycks_test",
                data_dir=cache_dir,
            )
            
            # Download and extract
            data_dir = dl_manager.download_and_extract(self.URL)
            print(f'Downloaded to: {data_dir}')

            # Load the CSV files for each split
            self.dataset = {}
            splits = {
                "test": f"data/test/{self.DATASET_NAME}_test.csv",
                "validation": f"data/val/{self.DATASET_NAME}_val.csv",
                "dev": f"data/dev/{self.DATASET_NAME}_dev.csv"
            }
            
            for split_name, file_path in splits.items():
                # Read CSV and convert to expected format
                full_path = os.path.join(data_dir, file_path)
                print(f"Loading {split_name} from {full_path}")
                
                examples = []
                try:
                    with open(full_path, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 6:  # Ensure row has enough columns
                                examples.append({
                                    "question": row[0],
                                    "choices": row[1:5],
                                    "answer": row[5] if isinstance(row[5], int) else ord(row[5]) - ord('A')
                                })
                    
                    # Convert to HF Dataset format
                    self.dataset[split_name] = datasets.Dataset.from_dict({
                        "question": [ex["question"] for ex in examples],
                        "choices": [ex["choices"] for ex in examples],
                        "answer": [ex["answer"] for ex in examples]
                    })
                    print(f"Successfully loaded {len(examples)} examples for {split_name}")
                    
                except FileNotFoundError:
                    print(f"Warning: Could not find file {full_path}")
                    continue
                
        except Exception as e:
            print(f"Error downloading/processing dataset: {e}")
            raise

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            Question: <prompt>
            Choices:
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Answer:
            """
            prompt = "Question: " + doc["question"] + "\nChoices:\n"
            prompt += "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
            )
            prompt += "Answer:"
            return prompt

        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": doc["choices"],
            "gold": doc["answer"] if isinstance(doc["answer"], int) else keys.index(doc["answer"])
        }

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't
        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))
        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

# class GeneralHendrycksTest(MultipleChoiceTask):
#     VERSION = 0
#     DATASET_PATH = "hendrycks_test"
#     DATASET_NAME = None

#     def __init__(self, subject):
#         self.DATASET_NAME = subject
#         super().__init__()

#     def has_training_docs(self):
#         return False

#     def has_validation_docs(self):
#         return True

#     def has_test_docs(self):
#         return True

#     def validation_docs(self):
#         return map(self._process_doc, self.dataset["validation"])

#     def test_docs(self):
#         return map(self._process_doc, self.dataset["test"])

#     def _process_doc(self, doc):
#         def format_example(doc, keys):
#             """
#             Question: <prompt>
#             Choices:
#             A. <choice1>
#             B. <choice2>
#             C. <choice3>
#             D. <choice4>
#             Answer:
#             """
#             prompt = "Question: " + doc["question"] + "\nChoices:\n"
#             prompt += "".join(
#                 [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
#             )
#             prompt += "Answer:"
#             return prompt

#         keys = ["A", "B", "C", "D"]
#         return {
#             "query": format_example(doc, keys),
#             "choices": doc["choices"],
#             "gold": keys.index(doc["answer"])
#             if isinstance(doc["answer"], str)
#             else doc["answer"],
#         }

#     def fewshot_examples(self, k, rnd):
#         # fewshot_examples is not just sampling from train_docs because dev is
#         # in the same distribution as val/test but auxiliary_train isn't

#         if self._fewshot_docs is None:
#             self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

#         return rnd.sample(list(self._fewshot_docs), k)

#     def doc_to_text(self, doc):
#         return doc["query"]

#     def should_decontaminate(self):
#         return True

#     def doc_to_decontamination_query(self, doc):
#         return doc["query"]
