# Official Ubuntu base image with Python 3.12
FROM ubuntu:24.04

WORKDIR /app

# System packages in ONE layer.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev curl ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 ships an externally-managed Python (PEP 668) whose pip comes from a .deb and has
# no RECORD file, so `pip install --upgrade pip` cannot uninstall it and dies with
#   "Cannot uninstall pip 24.0, RECORD file not found"
# — which failed every build here from 2026-06-15 onward. --break-system-packages does not help:
# it waives the PEP 668 guard, not the missing-RECORD uninstall. A venv owns its own pip, so the
# upgrade becomes an ordinary install and the whole class of problem disappears.
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv \
    PYTHONUNBUFFERED=1
RUN pip install --upgrade pip setuptools wheel

COPY . /app/
RUN pip install -e .

# Bake in the JEPA medium model the demo serves by default (LESNET_JEPA_HOME, default
# models/jepa). The app can self-heal by fetching it at runtime, but that would make the first
# request after a cold start pull 318 MB and very likely time out — so fetch it at build time.
RUN mkdir -p models/jepa && \
    curl -fsSL https://github.com/Thomasbehan/LesNet/releases/download/v5.0.0/lesnet-jepa-medium.tar.gz \
    | tar xz -C models/jepa && \
    test -f models/jepa/medium/jepa_config.json

# Construct the served predictor at BUILD time. A missing dependency or a bad artifact then fails
# the build loudly instead of surfacing as a 500 on the first user request in production.
RUN python -c "import lesnet.views.api as api; p = api._get_predictor(); \
print('predictor OK:', type(p).__name__, p.encoder.precision)"

# Run as a non-root user; it still owns /app so the runtime self-heal path can write there.
RUN useradd -ms /bin/bash appuser && chown -R appuser /app /opt/venv
USER appuser

EXPOSE 6543

CMD ["pserve", "production.ini"]
