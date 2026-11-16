import pytest

from utils.database import _password_hash, _verify_password


def test_password_hash_round_trip():
    encoded = _password_hash("Correct-Horse-Battery-42")
    assert _verify_password("Correct-Horse-Battery-42", encoded)
    assert not _verify_password("wrong-password", encoded)


def test_password_minimum_length():
    with pytest.raises(ValueError, match="12 characters"):
        _password_hash("short")
