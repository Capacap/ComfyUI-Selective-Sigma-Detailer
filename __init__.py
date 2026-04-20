from .selective_sigma_detailer import (
    SelectiveSigmaDetailerDeltaNode,
    SelectiveSigmaDetailerDeltaV2Node,
    SelectiveSigmaDetailerMaskPreviewNode,
)

NODE_CLASS_MAPPINGS = {
    "SelectiveSigmaDetailerDeltaNode": SelectiveSigmaDetailerDeltaNode,
    "SelectiveSigmaDetailerDeltaV2Node": SelectiveSigmaDetailerDeltaV2Node,
    "SelectiveSigmaDetailerMaskPreviewNode": SelectiveSigmaDetailerMaskPreviewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectiveSigmaDetailerDeltaNode": "Selective Sigma Detailer (Delta)",
    "SelectiveSigmaDetailerDeltaV2Node": "Selective Sigma Detailer (Delta V2)",
    "SelectiveSigmaDetailerMaskPreviewNode": "Selective Sigma Detailer Mask Preview",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
