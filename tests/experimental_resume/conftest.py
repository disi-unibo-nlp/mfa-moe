"""Private user-owned staging for cap regressions; no node-local or shared writes."""
from pathlib import Path
import tempfile
import pytest


@pytest.fixture
def workdir():
    root=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/codex-resume-tests')
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cap-',dir=root) as directory:
        yield Path(directory)
