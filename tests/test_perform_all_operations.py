from unittest.mock import Mock, patch

import pytest
from MDF_channel_plot import perform_plot_operation


@patch("tkinter.messagebox.showerror")
def test_successful_operation(mock_showerror):
    # Create a mock operation that succeeds
    mock_operation = Mock()
    mock_operation.__name__ = "test_operation"

    # Call the function
    perform_plot_operation(mock_operation, "arg1", "arg2", kwarg1="value1")

    # Assert that the operation was called with correct arguments
    mock_operation.assert_called_once_with("arg1", "arg2", kwarg1="value1")

    # Assert that no error message was shown
    mock_showerror.assert_not_called()


@patch("tkinter.messagebox.showerror")
def test_failing_operation(mock_showerror):
    # Create a mock operation that raises an exception
    mock_operation = Mock(side_effect=Exception("Test error"))
    mock_operation.__name__ = "test_operation"

    # Call the function
    perform_plot_operation(mock_operation, "arg1", "arg2", kwarg1="value1")

    # Assert that the operation was called with correct arguments
    mock_operation.assert_called_once_with("arg1", "arg2", kwarg1="value1")

    # Assert that an error message was shown
    mock_showerror.assert_called_once_with(
        "perform_plot_operation",
        "Failed to perform operation test_operation: Test error",
    )


@patch("tkinter.messagebox.showerror")
def test_operation_with_no_arguments(mock_showerror):
    # Create a mock operation that succeeds
    mock_operation = Mock()
    mock_operation.__name__ = "test_operation"

    # Call the function with no arguments
    perform_plot_operation(mock_operation)

    # Assert that the operation was called with no arguments
    mock_operation.assert_called_once_with()

    # Assert that no error message was shown
    mock_showerror.assert_not_called()


if __name__ == "__main__":
    pytest.main()
