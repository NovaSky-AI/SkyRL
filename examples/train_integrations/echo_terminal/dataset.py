import os
from typing import Any, List

import datasets
from loguru import logger
from transformers import PreTrainedTokenizerBase


class TerminalAgentTaskDataset:
    """Dataset for phitrain terminal-agent parquet files.

    Expected columns are:
    - prompt: chat messages
    - path: task identifier/path
    - task_binary: gzipped tar bytes for the Harbor task directory
    """

    def __init__(
        self,
        data_files: List[str | dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        prompt_key: str = "prompt",
        path_key: str = "path",
        task_binary_key: str = "task_binary",
        num_workers: int = 8,
        max_rows: int | None = None,
        max_rows_per_source: int | None = None,
        world_model_only_paths: list[str] | None = None,
    ) -> None:
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_key = prompt_key
        self.path_key = path_key
        self.task_binary_key = task_binary_key
        self.num_workers = num_workers
        self.max_rows = max_rows
        self.max_rows_per_source = max_rows_per_source
        self.world_model_only_paths = set(world_model_only_paths or [])
        self._read_files_and_tokenize()

    def _normalize_source(self, source: str | dict[str, Any]) -> tuple[str, bool, str]:
        if isinstance(source, str):
            return source, False, os.path.splitext(os.path.basename(source))[0]
        if not isinstance(source, dict):
            raise TypeError(f"Terminal-agent data source must be a path or dict, got {type(source)}")
        path = source.get("path") or source.get("file") or source.get("data")
        if not isinstance(path, str):
            raise ValueError("Terminal-agent data source dict must include a string 'path', 'file', or 'data'.")
        data_source = source.get("name") or source.get("data_source") or os.path.splitext(os.path.basename(path))[0]
        if not isinstance(data_source, str):
            raise ValueError("Terminal-agent data source dict 'name' or 'data_source' must be a string when provided.")
        return path, bool(source.get("world_model_only", False)), data_source

    def _load_file(self, source: str) -> datasets.Dataset:
        ext = os.path.splitext(source)[-1].lower()
        if ext == ".parquet":
            return datasets.load_dataset("parquet", data_files=source, keep_in_memory=True)["train"]
        if ext in [".json", ".jsonl"]:
            return datasets.load_dataset("json", data_files=source, keep_in_memory=True)["train"]
        raise ValueError(f"TerminalAgentTaskDataset expects parquet/json/jsonl data, got {source!r}")

    def _read_files_and_tokenize(self) -> None:
        loaded = []
        for source in self.data_files:
            path, world_model_only, data_source = self._normalize_source(source)
            dataset = self._load_file(path)
            if self.max_rows_per_source is not None:
                dataset = dataset.select(range(min(self.max_rows_per_source, len(dataset))))
            world_model_only = world_model_only or path in self.world_model_only_paths
            if world_model_only:
                dataset = dataset.map(lambda row: {**row, "_world_model_only": True, "_data_source": data_source})
            else:
                dataset = dataset.map(
                    lambda row: {
                        **row,
                        "_world_model_only": bool(row.get("_world_model_only", False)),
                        "_data_source": data_source,
                    }
                )
            loaded.append(dataset)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(loaded)
        if self.max_rows is not None:
            self.dataframe = self.dataframe.select(range(min(self.max_rows, len(self.dataframe))))
        logger.info(f"Loaded terminal-agent dataset with {len(self.dataframe)} rows")

        required = {self.prompt_key, self.path_key, self.task_binary_key}
        missing = required - set(self.dataframe.column_names)
        if missing:
            raise ValueError(f"Terminal-agent dataset is missing required columns: {sorted(missing)}")

        def _tokenize(row: dict[str, Any]) -> dict[str, Any]:
            row["prompt_token_ids"] = self.tokenizer.apply_chat_template(
                row[self.prompt_key],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            row["prompt_token_count"] = len(row["prompt_token_ids"])
            return row

        if self.num_workers > 1:
            import multiprocess

            multiprocess.set_start_method("spawn", force=True)

        self.dataframe = self.dataframe.map(
            _tokenize,
            num_proc=self.num_workers,
            desc="Tokenizing terminal-agent prompts",
        )
        self.dataframe = self.dataframe.filter(
            lambda row: row["prompt_token_count"] <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )
        logger.info(f"Filtered terminal-agent dataset size: {len(self.dataframe)}")

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe[index]
        env_extras = {
            "path": row[self.path_key],
            "task_binary": row[self.task_binary_key],
            "prompt_token_ids": row["prompt_token_ids"],
            "world_model_only": bool(row.get("_world_model_only", False)),
            "data_source": row.get("_data_source"),
        }
        return {
            "prompt": row[self.prompt_key],
            "env_class": "terminal_agent",
            "env_extras": env_extras,
            "uid": str(index),
        }

    def __len__(self) -> int:
        return len(self.dataframe)

    def collate_fn(self, item_list):
        return item_list
