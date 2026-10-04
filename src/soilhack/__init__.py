"""soilhack — shared library for the TabPFN-3.5 soil-health hackathon project.

Modules
-------
tabpfn_runner : explicit TabPFN-3.5 model factory (never the bare constructor)
metrics       : classification + regression metrics with interpretation helpers
data          : dataset loaders (verified sources only)
"""
__all__ = ["tabpfn_runner", "metrics", "data"]
