"""Downloads the named models' weights (default: all six) into $HF_HOME, because the GPU jobs run offline. Gemma is gated:
its licence must be accepted on the hub and HF_TOKEN set. A model that fails doesn't stop the others; the run ends with an
error."""
import os
import sys

from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from accuracy_tables.models import ADAPTERS  # noqa: E402

failed = []
for key in sys.argv[1:] or ADAPTERS:
    model_id = ADAPTERS[key].default_ids[0]
    try:
        print(f"{key}: {model_id} -> {snapshot_download(model_id)}", flush=True)
    except Exception as ex:
        failed.append(key)
        print(f"{key}: {model_id} FAILED: {type(ex).__name__}: {str(ex)[:300]}", flush=True)
if failed:
    sys.exit(f"not downloaded: {failed}")
