"""R&D tax-incentive reform dates extracted from policy documents.

Pipeline: data/policy_docs/<Country>/*.pdf|html → extract.py (Claude,
structured output) → verify.py (quote found on the page?) → reconcile.py
(documented vs dates derived from the OECD subsidy series).
"""
