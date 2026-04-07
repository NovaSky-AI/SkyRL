import array
import json
import logging
import mmap
import os
import struct
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class SHMBufferConfig:
    """Control block for Shared Memory Buffer."""
    FORMAT = "4sIIIII"
    SIZE = struct.calcsize(FORMAT)
    MAGIC = b"ATRP"
    VERSION = 1

class ZeroCopySHMBuffer:
    """
    High-performance circular buffer using multiprocessing.shared_memory.
    Eliminates serialization and HTTP overhead for trajectory transport.
    """

    def __init__(
        self,
        name: str,
        size: int = 1000,
        entry_size: int = 4096,
        instance_id_len: int = 64,
        metadata_len: int = 256,
        create: bool = False,
    ):
        self.name = name
        self.max_size = size
        self.entry_size = entry_size
        self.instance_id_len = instance_id_len
        self.metadata_len = metadata_len
        
        # Schema: [Score(8)|Len(4)|ID(len)|Rep(4)|Meta(len)|Tokens(size*4)]
        self.slot_size = (
            8 + 4 + instance_id_len + 4 + metadata_len + (entry_size * 4)
        )
        self.total_size = SHMBufferConfig.SIZE + (size * self.slot_size)
        
        try:
            if create:
                try:
                    shm = shared_memory.SharedMemory(name=name)
                    shm.unlink()
                except FileNotFoundError:
                    pass
                self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.total_size)
                self.buf = self.shm.buf
                self._init_control_block()
            else:
                self.shm = shared_memory.SharedMemory(name=name)
                self.buf = self.shm.buf
        except Exception as e:
            logger.error(f"SHM Init Failed: {e}")
            raise

    def _init_control_block(self):
        struct.pack_into(
            SHMBufferConfig.FORMAT,
            self.buf,
            0,
            SHMBufferConfig.MAGIC,
            SHMBufferConfig.VERSION,
            0, 0, self.max_size, self.entry_size,
        )

    def _get_control(self) -> Tuple[int, int, int, int]:
        magic, version, ridx, widx, msize, esize = struct.unpack_from(SHMBufferConfig.FORMAT, self.buf, 0)
        if magic != SHMBufferConfig.MAGIC:
            raise ValueError("Invalid Magic")
        return ridx, widx, msize, esize

    def _set_read_idx(self, idx: int):
        struct.pack_into("I", self.buf, 8, idx)

    def _set_write_idx(self, idx: int):
        struct.pack_into("I", self.buf, 12, idx)

    def write_trajectory(self, tokens: List[int], score: float, instance_id: str = "", rep_id: int = 0, metadata: Dict[str, Any] = None):
        ridx, widx, msize, esize = self._get_control()
        next_w = (widx + 1) % msize
        if next_w == ridx: return False

        offset = SHMBufferConfig.SIZE + (widx * self.slot_size)
        struct.pack_into("d", self.buf, offset, float(score))
        
        token_len = min(len(tokens), esize)
        struct.pack_into("i", self.buf, offset + 8, token_len)
        
        id_bytes = instance_id.encode('utf-8')[:self.instance_id_len]
        struct.pack_into(f"{self.instance_id_len}s", self.buf, offset + 12, id_bytes)
        
        struct.pack_into("i", self.buf, offset + 12 + self.instance_id_len, int(rep_id))
        
        meta = json.dumps(metadata or {}).encode('utf-8')[:self.metadata_len]
        struct.pack_into(f"{self.metadata_len}s", self.buf, offset + 12 + self.instance_id_len + 4, meta)
        
        t_off = offset + 12 + self.instance_id_len + 4 + self.metadata_len
        arr = np.array(tokens, dtype=np.int32)
        slot = np.ndarray((esize,), dtype=np.int32, buffer=self.buf, offset=t_off)
        slot[:token_len] = arr[:token_len]
        if token_len < esize: slot[token_len:] = 0
            
        self._set_write_idx(next_w)
        return True

    def read_next(self) -> Optional[Dict[str, Any]]:
        ridx, widx, msize, esize = self._get_control()
        if ridx == widx: return None
            
        offset = SHMBufferConfig.SIZE + (ridx * self.slot_size)
        score = struct.unpack_from("d", self.buf, offset)[0]
        token_len = min(struct.unpack_from("i", self.buf, offset + 8)[0], esize)
        
        id_b = struct.unpack_from(f"{self.instance_id_len}s", self.buf, offset + 12)[0]
        inst_id = id_b.decode('utf-8', errors='ignore').strip('\x00')
        
        rep_id = struct.unpack_from("i", self.buf, offset + 12 + self.instance_id_len)[0]
        
        meta_b = struct.unpack_from(f"{self.metadata_len}s", self.buf, offset + 12 + self.instance_id_len + 4)[0]
        try:
            meta = json.loads(meta_b.decode('utf-8', errors='ignore').strip('\x00'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            meta = {}
            
        t_off = offset + 12 + self.instance_id_len + 4 + self.metadata_len
        tokens = np.ndarray((token_len,), dtype=np.int32, buffer=self.buf, offset=t_off)
        
        self._set_read_idx((ridx + 1) % msize)
        return {
            "tokens": tokens.tolist(),
            "score": score,
            "instance_id": inst_id,
            "repetition_id": rep_id,
            "metadata": meta
        }

    def close(self, unlink: bool = False):
        self.shm.close()
        if unlink: self.shm.unlink()
