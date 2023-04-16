import os
import subprocess
from pyramid.view import view_config

log_dir = 'logs'


@view_config(route_name='tensorboard', renderer='string')
def tensorboard_view(request):
    if not os.path.exists(log_dir):
        return "Log directory not found. Please train the model first."

    # Start TensorBoard as a background process
    command = f"tensorboard --logdir {log_dir} --host 0.0.0.0 --port 6006"
    subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return "TensorBoard is running on http://localhost:6006"
