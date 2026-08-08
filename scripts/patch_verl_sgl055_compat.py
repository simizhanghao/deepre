#!/usr/bin/env python3
"""Patch veRL async_sglang_server for sglang 0.5.5 (verlai/verl:sgl055.latest).

Latest veRL imports ContinueGenerationReqInput / PauseGenerationReqInput and
calls pause/continue with those objects. SGLang 0.5.5 has the methods but:
  - those request classes are missing
  - pause_generation()/continue_generation() take no arguments
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

DEFAULT = Path("/workspace/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py")

OLD_IMPORT = """from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    GenerateReqInput,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)"""

NEW_IMPORT = """try:
    from sglang.srt.managers.io_struct import (
        ContinueGenerationReqInput,
        GenerateReqInput,
        PauseGenerationReqInput,
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
    )
except ImportError:  # sglang<=0.5.5 (verlai/verl:sgl055.latest)
    from dataclasses import dataclass

    from sglang.srt.managers.io_struct import (
        GenerateReqInput,
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
    )

    @dataclass
    class ContinueGenerationReqInput:
        pass

    @dataclass
    class PauseGenerationReqInput:
        mode: str = "abort"

    _SGLANG_PAUSE_CONTINUE_NOARGS = True
else:
    _SGLANG_PAUSE_CONTINUE_NOARGS = False"""

OLD_METHODS = """    async def abort_all_requests(self):
        if self.node_rank != 0:
            return
        await self.tokenizer_manager.pause_generation(PauseGenerationReqInput(mode="abort"))

    async def resume_generation(self):
        if self.node_rank != 0:
            return
        await self.tokenizer_manager.continue_generation(ContinueGenerationReqInput())"""

NEW_METHODS = """    async def abort_all_requests(self):
        if self.node_rank != 0:
            return
        # sglang 0.5.5: pause_generation() takes no args; newer: PauseGenerationReqInput
        if globals().get("_SGLANG_PAUSE_CONTINUE_NOARGS", False):
            await self.tokenizer_manager.pause_generation()
        else:
            await self.tokenizer_manager.pause_generation(PauseGenerationReqInput(mode="abort"))

    async def resume_generation(self):
        if self.node_rank != 0:
            return
        if globals().get("_SGLANG_PAUSE_CONTINUE_NOARGS", False):
            await self.tokenizer_manager.continue_generation()
        else:
            await self.tokenizer_manager.continue_generation(ContinueGenerationReqInput())"""


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        raise SystemExit(f"missing {path}")
    text = path.read_text(encoding="utf-8")
    if "_SGLANG_PAUSE_CONTINUE_NOARGS" in text:
        print(f"[skip] already patched: {path}")
        return
    if OLD_IMPORT not in text:
        raise SystemExit("import block not found; verl source changed")
    if OLD_METHODS not in text:
        raise SystemExit("abort/resume block not found; verl source changed")
    bak = path.with_suffix(path.suffix + ".bak_sgl055")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[backup] {bak}")
    text = text.replace(OLD_IMPORT, NEW_IMPORT, 1).replace(OLD_METHODS, NEW_METHODS, 1)
    path.write_text(text, encoding="utf-8")
    print(f"[ok] patched {path}")


if __name__ == "__main__":
    main()
