from .selective_sigma_detailer import SelectiveSigmaDetailerNode

NODE_CLASS_MAPPINGS = {
    "SelectiveSigmaDetailerNode": SelectiveSigmaDetailerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectiveSigmaDetailerNode": "Selective Sigma Detailer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
