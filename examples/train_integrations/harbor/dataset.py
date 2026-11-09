from loguru import logger
from typing import List, Tuple
from pathlib import Path


class HarborTaskDataset:
    """
    A dataset that loads Harbor task data from direct file/directory paths.
    Each dataset item is a path to a task directory. Its ``data_source`` is the name of the ``data_files`` entry
    it was found under, so eval reports one score per directory given rather than one per task; the task's own
    path is in ``env_extras["task_path"]``.
    """

    def __init__(
        self,
        data_files: List[str],
    ):
        """
        Initialize the HarborTaskDataset.

        Args:
            data_files: List of direct file/directory paths pointing to Harbor task data
        """
        self.data_files = data_files

        # Load all data files
        self.task_paths, self.data_sources = self._load_data_files()

        logger.info(f"HarborTaskDataset initialized with {len(self.task_paths)} task paths")

    def _load_data_files(self) -> Tuple[List[Path], List[str]]:
        """Load all data files from direct paths; return the task paths and, for each, its data source."""
        task_paths = []
        data_sources = []

        for data_source in self.data_files:
            source_path = Path(data_source)

            if not source_path.exists():
                logger.warning(f"Path does not exist: {data_source}")
                continue

            logger.info(f"Loading data from: {data_source}")

            # If the path is a directory, find all valid task subdirectories
            if source_path.is_dir():
                # Look for task subdirectories and validate them
                all_dirs = [d for d in source_path.iterdir() if d.is_dir()]
                valid_task_dirs = [d for d in all_dirs if self._is_valid_task_directory(d)]

                if valid_task_dirs:
                    task_paths.extend(valid_task_dirs)
                    data_sources.extend([source_path.name] * len(valid_task_dirs))
                    logger.info(
                        f"Found {len(valid_task_dirs)} valid task directories out of {len(all_dirs)} total directories"
                    )
                elif self._is_valid_task_directory(source_path):
                    # If no subdirectories but the main directory is valid, treat it as a task
                    task_paths.append(source_path)
                    data_sources.append(source_path.name)
                    logger.info("Using main directory as valid task")
                else:
                    logger.warning(f"No valid task directories found in {source_path}")
            else:
                # If it's a file, treat it as a single task (files can't be valid task directories)
                logger.warning(f"File {source_path} cannot be a valid task directory (missing instruction.md)")

        return task_paths, data_sources

    def _is_valid_task_directory(self, task_path: Path) -> bool:
        """Check if a directory is a valid task directory (has instruction.md file)."""
        if not task_path.is_dir():
            return False

        instruction_file = task_path / "instruction.md"
        return instruction_file.exists() and instruction_file.is_file()

    def __getitem__(self, index: int) -> dict:
        """Get a task path by index as a dictionary with 'prompt', 'env_class', and 'env_extras' keys."""
        if index >= len(self.task_paths):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.task_paths)}")
        return self._item(index)

    def __len__(self) -> int:
        """Return the number of tasks in the dataset."""
        return len(self.task_paths)

    def __iter__(self):
        """Iterate over all task paths as dictionaries."""
        for index in range(len(self.task_paths)):
            yield self._item(index)

    def _item(self, index: int) -> dict:
        task_path = str(self.task_paths[index])
        return {
            "prompt": task_path,
            "env_class": None,
            "env_extras": {"data_source": self.data_sources[index], "task_path": task_path},
            "uid": str(index),
        }

    def get_task_paths(self) -> List[Path]:
        """Return all task paths as a list."""
        return self.task_paths.copy()

    def collate_fn(self, item_list):
        """Collate function for batching task dictionaries."""
        return item_list
