from .selective_sigma_detailer import (
    SelectiveSigmaDetailerDeltaV2Node,
    SelectiveSigmaDetailerMaskPreviewNode,
)

NODE_CLASS_MAPPINGS = {
    "SelectiveSigmaDetailerDeltaV2Node": SelectiveSigmaDetailerDeltaV2Node,
    "SelectiveSigmaDetailerMaskPreviewNode": SelectiveSigmaDetailerMaskPreviewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectiveSigmaDetailerDeltaV2Node": "Selective Sigma Detailer",
    "SelectiveSigmaDetailerMaskPreviewNode": "Selective Sigma Detailer Mask Preview",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
