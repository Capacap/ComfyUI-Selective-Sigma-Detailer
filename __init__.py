from .selective_sigma_detailer import (
    SelectiveSigmaDetailerDebugNode,
    SelectiveSigmaDetailerNode,
    SelectiveSigmaDetailerMaskPreviewNode,
)

NODE_CLASS_MAPPINGS = {
    "SelectiveSigmaDetailerNode": SelectiveSigmaDetailerNode,
    "SelectiveSigmaDetailerDebugNode": SelectiveSigmaDetailerDebugNode,
    "SelectiveSigmaDetailerMaskPreviewNode": SelectiveSigmaDetailerMaskPreviewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectiveSigmaDetailerNode": "Selective Sigma Detailer",
    "SelectiveSigmaDetailerDebugNode": "Selective Sigma Detailer (Debug)",
    "SelectiveSigmaDetailerMaskPreviewNode": "Selective Sigma Detailer (Debug Preview)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
