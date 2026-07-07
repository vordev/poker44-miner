"""Poker44 miner training toolkit.

Modules:
- benchmark_client: pull + cache + parse the public training-benchmark API.
- features:          chunk -> fixed-length feature vector (shared train & serve).
- train:             train a model and evaluate with the *exact* subnet reward.
- serve:             load a trained model and score chunks inside the miner.

Design rule: `features.extract_features` is imported by BOTH training and the
live miner so there is zero train/serve skew. The benchmark API already returns
hands in the same sanitized, miner-visible form the validator serves.
"""
