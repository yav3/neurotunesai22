"""Cross-repo bridge — connects NeuroTunesAI to other repos in the portfolio.

Architecture:
  NeuroTunesAI (math kernels) ←→ merlinaineuralplus2 (evidence engine, GEO data)
  NeuroTunesAI (math kernels) ←→ neurotunescopy4privatesharing (React product)
  NeuroTunesAI (math kernels) ←→ neurotunes (Flutter app)

The bridge translates between:
  - NeuroTunesAI's Problem/Answer types
  - Merlin's evidence records, GEO data envelopes, engine results
  - Product-surface audio analysis features (VAD, EIF, BPM)
"""
from models.bridge.merlin_bridge import MerlinBridge
from models.bridge.product_bridge import ProductBridge
from models.bridge.geo_adapter import GEOAdapter

__all__ = ["MerlinBridge", "ProductBridge", "GEOAdapter"]
