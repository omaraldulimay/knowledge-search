import numpy as np

from ks.embed import HashEmbedder


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder()
    a = e.embed(["Urlaub beantragen", "VPN"])
    b = HashEmbedder().embed(["Urlaub beantragen", "VPN"])
    assert a.shape == (2, e.dim)
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_similar_texts_are_closer():
    e = HashEmbedder()
    q, close, far = e.embed(["Urlaub beantragen", "Urlaub wird beantragt", "Zollanmeldung ATLAS"])
    assert q @ close > q @ far or q @ far == 0
