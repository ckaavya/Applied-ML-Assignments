from homework2_rent import score_rent

def test_rent():

    p = score_rent()
    assert p >= 0.59
