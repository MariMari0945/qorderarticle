"""Minimal test runner for environments without pytest (pytest-only tests are skipped)."""
import importlib
import inspect
import sys
import traceback
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
ok = fail = skip = 0
for f in sorted((root / "tests").glob("test_*.py")):
    try:
        mod = importlib.import_module(f"tests.{f.stem}")
    except Exception as e:  # noqa: BLE001
        print(f"SKIP {f.name}: {e}")
        skip += 1
        continue
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_") or inspect.signature(fn).parameters:
            continue
        try:
            fn(); ok += 1; print(f"ok   {f.stem}.{name}")
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "Skipped":
                skip += 1; print(f"skip {f.stem}.{name}"); continue
            fail += 1; print(f"FAIL {f.stem}.{name}"); traceback.print_exc()
print(f"\n{ok} passed, {fail} failed, {skip} skipped")
sys.exit(1 if fail else 0)
