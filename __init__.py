from .selective_sigma_detailer import (
    SelectiveSigmaDetailerDeltaNode,
    SelectiveSigmaDetailerDynamicVarianceNode,
    SelectiveSigmaDetailerEdgesNode,
    SelectiveSigmaDetailerMaskPreviewNode,
    SelectiveSigmaDetailerNode,
)

NODE_CLASS_MAPPINGS = {
    "SelectiveSigmaDetailerNode": SelectiveSigmaDetailerNode,
    "SelectiveSigmaDetailerDynamicVarianceNode": SelectiveSigmaDetailerDynamicVarianceNode,
    "SelectiveSigmaDetailerDeltaNode": SelectiveSigmaDetailerDeltaNode,
    "SelectiveSigmaDetailerEdgesNode": SelectiveSigmaDetailerEdgesNode,
    "SelectiveSigmaDetailerMaskPreviewNode": SelectiveSigmaDetailerMaskPreviewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectiveSigmaDetailerNode": "Selective Sigma Detailer (Variance Snapshot)",
    "SelectiveSigmaDetailerDynamicVarianceNode": "Selective Sigma Detailer (Dynamic Variance)",
    "SelectiveSigmaDetailerDeltaNode": "Selective Sigma Detailer (Delta)",
    "SelectiveSigmaDetailerEdgesNode": "Selective Sigma Detailer (Edges)",
    "SelectiveSigmaDetailerMaskPreviewNode": "Selective Sigma Detailer Mask Preview",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
