def test_sdk_import():
    """Test if the SDK can be imported without any syntax errors."""
    from lexsi_sdk import xai
    assert hasattr(xai, 'login')
