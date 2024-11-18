# from datasets import load_dataset
# import sys

# def test_mmlu_loading():
#     try:
#         # Test single subject
#         dataset = load_dataset('hendrycks_test', 'abstract_algebra')
#         print("Successfully loaded abstract_algebra:")
#         print(dataset)
        
#         # Test another subject
#         dataset = load_dataset("hendrycks_test", "anatomy")
#         print("\nSuccessfully loaded anatomy:")
#         print(dataset)
        
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         print(f"Error type: {type(e)}")
#         print(f"Python path: {sys.path}")

# if __name__ == "__main__":
#     test_mmlu_loading() 

from lm_eval.tasks.hendrycks_test import create_task

def test_mmlu_loading():
    try:
        # Test single subject
        print("Testing abstract_algebra:")
        task = create_task("abstract_algebra")()
        print("Dataset splits:", task.dataset.keys())
        print("Validation sample:", next(task.validation_docs()))
        
        # Test another subject
        print("\nTesting anatomy:")
        task = create_task("anatomy")()
        print("Dataset splits:", task.dataset.keys())
        print("Validation sample:", next(task.validation_docs()))
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mmlu_loading()