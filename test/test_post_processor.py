import pandas as pd

from src.features.post_processor import post_process


def test_post_process_have_same_amount_of_rows_as_input():
    # Given
    data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # When
    processed_data = post_process(data)
    # Then
    assert len(processed_data) == len(data)
