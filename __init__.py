from .selective_sigma_detailer import (
    SelectiveSigmaDetailerDebugNode,
    SelectiveSigmaDetailerDeltaV2Node,
    SelectiveSigmaDetailerMaskPreviewNode,
    SelectiveSigmaDetailerPatchModelNode,
)

NODE_CLASS_MAPPINGS = {
    "SelectiveSigmaDetailerDeltaV2Node": SelectiveSigmaDetailerDeltaV2Node,
    "SelectiveSigmaDetailerPatchModelNode": SelectiveSigmaDetailerPatchModelNode,
    "SelectiveSigmaDetailerDebugNode": SelectiveSigmaDetailerDebugNode,
    "SelectiveSigmaDetailerMaskPreviewNode": SelectiveSigmaDetailerMaskPreviewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectiveSigmaDetailerDeltaV2Node": "Selective Sigma Detailer",
    "SelectiveSigmaDetailerPatchModelNode": "Selective Sigma Detailer (Model Patch)",
    "SelectiveSigmaDetailerDebugNode": "Selective Sigma Detailer (Debug)",
    "SelectiveSigmaDetailerMaskPreviewNode": "Selective Sigma Detailer (Debug Preview)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
