import os
from pyramid import testing
from skinvestigatorai.views.tensorboard_view import tensorboard_view, log_dir


def test_tensorboard_view():
    if os.path.exists(log_dir):
        os.system(f"mv {log_dir} {log_dir}_backup")

    request = testing.DummyRequest()
    response = tensorboard_view(request)
    assert response == \
           "Log directory not found. Please train the model first." or "TensorBoard is running on http://localhost:6006"

    if os.path.exists(f"{log_dir}_backup"):
        os.system(f"mv {log_dir}_backup {log_dir}")
