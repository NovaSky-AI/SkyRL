"""ThunderAgent-local Harbor task dataset with curated-subset and dedup support."""

import json
from pathlib import Path
from typing import List, Optional


class ThunderAgentHarborDataset:
    """
    A dataset that loads Harbor task data from direct file/directory paths.
    Supports max_tasks limiting, MANIFEST.json curated subsets, stable path-based
    UIDs, and duplicate filtering.
    """

    def __init__(
        self,
        data_files: List[str],
        max_tasks: Optional[int] = None,
    ):
        self.data_files = data_files
        self.task_paths = self._load_data_files()

        if not self.task_paths:
            raise ValueError(
                f"ThunderAgentHarborDataset resolved zero task directories from "
                f"data_files={data_files!r}. Each entry must be a directory of "
                f"task subdirectories (each containing instruction.md), a single "
                f"task directory, or a directory with a curated MANIFEST.json."
            )

        if max_tasks is not None and max_tasks < len(self.task_paths):
            self.task_paths = self.task_paths[:max_tasks]

    @staticmethod
    def _canonicalize_task_path(task_path: Path) -> Path:
        return task_path.expanduser().resolve()

    @classmethod
    def _make_uid(cls, task_path: Path) -> str:
        return str(cls._canonicalize_task_path(task_path))

    def _load_manifest_task_paths(self, manifest_path: Path) -> List[Path]:
        """Resolve task directories listed in a curated-subset MANIFEST.json."""
        try:
            data = json.loads(manifest_path.read_text())
        except Exception:
            return []

        tasks = data.get("tasks")
        if not isinstance(tasks, dict):
            return []

        data_root = manifest_path.parent.parent
        resolved_paths = []
        for bucket_tasks in tasks.values():
            if not isinstance(bucket_tasks, list):
                continue
            for rel_task_path in bucket_tasks:
                task_path = (data_root / rel_task_path).resolve()
                if self._is_valid_task_directory(task_path):
                    resolved_paths.append(task_path)

        return resolved_paths

    def _load_data_files(self) -> List[Path]:
        """Load all data files from direct paths and return list of task paths."""
        task_paths = []
        seen_uids = set()

        for data_source in self.data_files:
            source_path = Path(data_source)

            if not source_path.exists():
                continue

            if source_path.is_dir():
                all_dirs = sorted(d for d in source_path.iterdir() if d.is_dir())
                valid_task_dirs = [d for d in all_dirs if self._is_valid_task_directory(d)]

                if valid_task_dirs:
                    for task_dir in valid_task_dirs:
                        uid = self._make_uid(task_dir)
                        if uid in seen_uids:
                            continue
                        task_paths.append(self._canonicalize_task_path(task_dir))
                        seen_uids.add(uid)
                elif self._is_valid_task_directory(source_path):
                    uid = self._make_uid(source_path)
                    if uid not in seen_uids:
                        task_paths.append(self._canonicalize_task_path(source_path))
                        seen_uids.add(uid)
                elif (source_path / "MANIFEST.json").is_file():
                    manifest_task_dirs = self._load_manifest_task_paths(source_path / "MANIFEST.json")
                    for task_dir in manifest_task_dirs:
                        uid = self._make_uid(task_dir)
                        if uid in seen_uids:
                            continue
                        task_paths.append(self._canonicalize_task_path(task_dir))
                        seen_uids.add(uid)
            else:
                # Files cannot be valid task directories
                pass

        return task_paths

    def _is_valid_task_directory(self, task_path: Path) -> bool:
        """Check if a directory is a valid task directory (has instruction.md file)."""
        if not task_path.is_dir():
            return False
        instruction_file = task_path / "instruction.md"
        return instruction_file.exists() and instruction_file.is_file()

    def __getitem__(self, index: int) -> dict:
        if index >= len(self.task_paths):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.task_paths)}")
        return {
            "prompt": str(self.task_paths[index]),
            "env_class": None,
            "env_extras": {"data_source": str(self.task_paths[index])},
            "uid": self._make_uid(self.task_paths[index]),
        }

    def __len__(self) -> int:
        return len(self.task_paths)

    def __iter__(self):
        for task_path in self.task_paths:
            yield {
                "prompt": str(task_path),
                "env_class": None,
                "env_extras": {"data_source": str(task_path)},
                "uid": self._make_uid(task_path),
            }

    def get_task_paths(self) -> List[Path]:
        return self.task_paths.copy()

    def collate_fn(self, item_list):
        return item_list
