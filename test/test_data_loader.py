from src.data.data_loader import DataLoader, create_data_loader


def test_create_data_loader():
    loader = create_data_loader()
    assert isinstance(loader, DataLoader)
