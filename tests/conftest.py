import pytest

@pytest.fixture
def anyio_backend():
    # Restrict anyio to asyncio backend; avoid requiring trio dependency
    return 'asyncio'
