import logging
from typing import List, Any, Union
from uuid import UUID, uuid4
import dspy
import json
import pickle
import os
import base64
import zlib
from pydantic import BaseModel
from .utils import get_benchmark_data

logger = logging.getLogger(__name__)


def _has_test_type(tests, test_type):
    """Check if any test in the test list has 'testtype' set to 'type'.
    
    Args:
        tests: Can be a JSON string, list, or dict
        test_type: The test type to check for (e.g., 'stdin', 'functional')
    """
    # Handle different input types
    if isinstance(tests, str):
        try:
            test_list = json.loads(tests)
        except json.JSONDecodeError:
            return False
    elif isinstance(tests, list):
        test_list = tests
    elif isinstance(tests, dict):
        # If it's a dict, check if it has a list value
        test_list = tests.get("tests", tests.get("public", tests.get("private", [])))
        if not isinstance(test_list, list):
            return False
    else:
        return False
    
    for test in test_list:
        if isinstance(test, dict) and test.get("testtype") == test_type:
            return True
    return False


def _translate_private_test_cases(encoded_data):
    """Decode and decompress private test cases."""
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)


def _update_dataset_in_place(dataset):
    """Helper function to translate the test cases."""
    for i, entry in enumerate(dataset):
        tests = entry.get("tests")
        if tests is None:
            continue
        
        # Check if already decoded (is a list/dict)
        if isinstance(tests, (list, dict)):
            # Already decoded, no need to decode
            continue
        
        # Try to decode if it's a string (might be encoded)
        if isinstance(tests, str):
            try:
                # First try to parse as JSON (might already be JSON string)
                try:
                    decoded = json.loads(tests)
                    entry["tests"] = decoded
                    continue
                except json.JSONDecodeError:
                    pass
                
                # If not JSON, try to decode as base64/zlib encoded
                decoded_tests = _translate_private_test_cases(tests)
                entry["tests"] = decoded_tests
            except Exception as e:
                logger.warning(f"Failed to decode test cases for entry {i}: {e}")
                # Keep original if decoding fails


def _map_to_dspy_example(row):
    """Map a dataset row to a dspy example format.
    
    Returns only:
    - prompt: from problem/question_content
    - tests: from tests
    """
    return {
        "prompt": row["question_content"],
        "tests": row["tests"],
    }


class LCBExample(BaseModel):
    """Pydantic model representing a DSPy example with UUID."""
    uuid: UUID
    prompt: str
    tests: Union[List[dict], dict, Any]
    
    class Config:
        arbitrary_types_allowed = True


class DSPyDataset:
    """
    A dataset that loads Live Code Bench data and converts it to DSPy examples.
    """

    def __init__(
        self,
        benchmark_name: str,
        max_num_examples: int = None,
    ):
        """
        Initialize the DSPyDataset.

        Args:
            data_file: JSON file path (e.g., "/path/to/livecodebench.json")
            max_num_examples: Maximum number of examples to return. If None, returns all examples.
        """
        
        self.benchmark_name = benchmark_name
        self.train_set, self.test_set = get_benchmark_data(benchmark_name)
        self.examples = self.train_set[:max_num_examples]

        # self.data_file = data_file
        # self.max_num_examples = max_num_examples
        # print('loading dspy dataset...')
        # pkl_path = "/home/ray/data/lcb/live_code_bench_dataset_test.pkl"
        # with open(pkl_path, "rb") as f:
        #     examples = pickle.load(f)
        # train_set, test_set = examples[:400], examples[400:]
        # self.examples = train_set
        # print('done loading dspy dataset')

        logger.info(f"DSPyDataset initialized with {len(self.examples)} examples")

    def _load_dataset(self) -> List[dspy.Example]:
        """Load dataset from JSON file."""
        if not os.path.exists(self.data_file):
            logger.warning(f"JSON file does not exist: {self.data_file}")
            return []

        if not self.data_file.endswith(".json"):
            logger.warning(f"File is not a JSON file: {self.data_file}")
            return []

        logger.info(f"Loading dataset from JSON file: {self.data_file}")
        examples = self._load_json_file(self.data_file, "train")

        # Apply limit if specified
        if self.max_num_examples is not None and len(examples) > self.max_num_examples:
            examples = examples[:self.max_num_examples]
            logger.info(f"Limited dataset to {self.max_num_examples} examples")

        return examples

    def _load_json_file(self, json_file_path: str, split_name: str = None) -> List[dspy.Example]:
        """Load and process a JSON file into DSPy examples.
        
        Args:
            json_file_path: Path to the JSON file
            split_name: Name of the split (e.g., "train", "test") for logging purposes
        """
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {json_file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading JSON file {json_file_path}: {e}")
            return []

        # Convert to list if it's a single dict
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            logger.error(f"JSON file {json_file_path} does not contain a list or dict")
            return []

        # Process entries to match the expected format
        processed_entries = []
        for idx, entry in enumerate(data):
            processed_entry = self._process_json_entry(entry, idx)
            if processed_entry is not None:
                processed_entries.append(processed_entry)

        # Decode test cases if they are encoded
        _update_dataset_in_place(processed_entries)

        # Create dspy.Example objects with UUIDs
        examples = []
        for row in processed_entries:
            example = dspy.Example(**_map_to_dspy_example(row)).with_inputs(
                "prompt", "tests"
            )
            # Add UUID to the example
            example.uuid = uuid4()
            # Add is_stdin to the example
            example.is_stdin = _has_test_type(example.tests, "stdin")
            
            examples.append(example)
        return examples

    def _process_json_entry(self, entry: dict, index: int = None) -> dict:
        """Process a single JSON entry to match the expected format.
        
        Based on investigation, the JSON files have:
        - problem: str (the question) -> question_content
        - tests: list of dicts with 'input', 'output', 'testtype' keys -> tests
        
        Returns dict with only:
        - question_content: for prompt
        - tests: for tests
        """
        # Check if entry already has the expected format
        if "question_content" in entry and "tests" in entry:
            # Already in the expected format
            return entry
        
        processed = {}
        
        # Map question/problem -> question_content
        if "problem" in entry:
            processed["question_content"] = entry["problem"]
        elif "question_content" in entry:
            processed["question_content"] = entry["question_content"]
        else:
            logger.warning(f"Entry missing 'problem' or 'question_content': {entry.keys()}")
            return None
        
        # Map tests -> tests
        # The tests field is a list of dicts with 'input', 'output', 'testtype'
        if "tests" in entry:
            tests = entry["tests"]
            # If it's a string, try to parse as JSON
            if isinstance(tests, str):
                try:
                    tests = json.loads(tests)
                except json.JSONDecodeError:
                    # If it's not valid JSON, might be encoded - will be handled by _update_dataset_in_place
                    processed["tests"] = tests
                    return processed
            
            # Store tests (already a list, no encoding needed)
            processed["tests"] = tests
        else:
            logger.warning(f"Entry missing 'tests': {entry.keys()}")
            return None
        
        return processed

    def __getitem__(self, index: int) -> LCBExample:
        """Get a DSPy example by index.
        
        Returns a dspy.Example object that conforms to the LCBExample Pydantic model structure.
        The example will have a uuid attribute in addition to prompt and tests.
        """
        if index >= len(self.examples):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.examples)}")
        return {
                "prompt": self.examples[index],
                "env_class": None,
                "env_extras": None,
                "uid": str(index),
            }

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def __iter__(self):
        """Iterate over all DSPy examples."""
        for index, example in enumerate(self.examples):
            yield {
                "prompt": example,
                "env_class": None,
                "env_extras": None,
                "uid": str(index),
            }
    
    def collate_fn(self, item_list):
        return item_list
