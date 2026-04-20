from .selective_sigma_detailer import (
    SelectiveSigmaDetailerDeltaV2Node,
    SelectiveSigmaDetailerPatchModelNode,
    SelectiveSigmaDetailerMaskPreviewNode,
)

NODE_CLASS_MAPPINGS = {
    "SelectiveSigmaDetailerDeltaV2Node": SelectiveSigmaDetailerDeltaV2Node,
    "SelectiveSigmaDetailerPatchModelNode": SelectiveSigmaDetailerPatchModelNode,
    "SelectiveSigmaDetailerMaskPreviewNode": SelectiveSigmaDetailerMaskPreviewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectiveSigmaDetailerDeltaV2Node": "Selective Sigma Detailer",
    "SelectiveSigmaDetailerPatchModelNode": "Selective Sigma Detailer (Model Patch)",
    "SelectiveSigmaDetailerMaskPreviewNode": "Selective Sigma Detailer Mask Preview",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
