import json, os
DEFAULT_ANCHORS_PATH = os.path.expanduser("~/.pixelart_maker/anchors.json")
PACKAGE_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "assets", "anchors_mediterranean.json")

def load_anchors():
    # Prefer user-defined anchors if exist, else packaged Mediterranean
    path = DEFAULT_ANCHORS_PATH if os.path.exists(DEFAULT_ANCHORS_PATH) else PACKAGE_DEFAULT
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [tuple(map(int, rgb)) for rgb in data.get("rgb", [])]
    except Exception:
        # minimal safe set
        return [(252,252,252),(8,8,8),(28,110,216),(19,118,204),(100,180,255),(62,115,65),(216,179,138)]
