"""Node implementations for ComfyUI-TrajectoryForcing.

Import order matters: `nodes()` is called from the extension's
`comfy_entrypoint`, and the node modules pull in `comfy_api`, so nothing here is
importable outside a ComfyUI process. The building blocks under it
(`data`, `tokens`, `render`, `locate`) are, and the tests use them directly.
"""
from __future__ import annotations


def nodes() -> list[type]:
    from .nodes_edit import TFFeatureEdit, TFResumeFromLevel, TFShapeEdit
    from .nodes_io import TFLoadLevels, TFSaveLevels
    from .nodes_pipeline import (
        TFDecode,
        TFGenerate,
        TFImageNetClass,
        TFLatentPreview,
        TFLevelsInfo,
        TFLoadPipeline,
    )
    from .nodes_regions import (
        TFLevelCanvas,
        TFRegionMap,
        TFTokensCombine,
        TFTokensFromCoords,
        TFTokensFromMask,
        TFTokensPreview,
    )

    return [
        TFLoadPipeline,
        TFImageNetClass,
        TFGenerate,
        TFDecode,
        TFLatentPreview,
        TFLevelsInfo,
        TFLevelCanvas,
        TFRegionMap,
        TFTokensFromMask,
        TFTokensFromCoords,
        TFTokensCombine,
        TFTokensPreview,
        TFFeatureEdit,
        TFShapeEdit,
        TFResumeFromLevel,
        TFSaveLevels,
        TFLoadLevels,
    ]
