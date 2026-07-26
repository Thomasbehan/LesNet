"""Live training dashboard — a nicer read on a run than TensorBoard, with no dependencies.

Serves `lesnet/jepa/dashboard.html` plus a `data.json` built by tailing the run's progress.jsonl
(written by lesnet.jepa.progress.ProgressWriter). Pure stdlib: no TensorBoard, no protobufs, and
nothing to install on the training box.

    python commands/train_dashboard.py --artifacts artifacts/family/small --port 6600
    python commands/train_dashboard.py --artifacts artifacts/jepa_large   # then open the URL

Watches multiple runs if given a parent directory containing several artifact dirs.
"""
import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1] / 'lesnet' / 'jepa' / 'dashboard.html'
MAX_POINTS = 600          # downsample so a long run stays snappy in the browser


def _read_progress(path, max_points=MAX_POINTS):
    """Parse progress.jsonl into {run, steps, epochs, finished}, tolerating a half-written line."""
    run, steps, epochs, finished = {}, [], [], None
    if not path.exists():
        return {'run': run, 'steps': steps, 'epochs': epochs, 'finished': finished}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:      # the trainer may be mid-write on the last line
                continue
            kind = record.get('kind')
            if kind == 'run':
                run = record
            elif kind == 'step':
                steps.append(record)
            elif kind == 'epoch':
                epochs.append(record)
            elif kind == 'finish':
                finished = record.get('reason')
    if len(steps) > max_points:               # keep the newest detail, thin the history
        stride = len(steps) // max_points + 1
        steps = steps[::stride] + steps[-1:]
    return {'run': run, 'steps': steps, 'epochs': epochs, 'finished': finished}


def _resolve(artifacts):
    """The artifacts dir itself if it has a progress.jsonl, else the most recent child that does."""
    artifacts = Path(artifacts)
    if (artifacts / 'progress.jsonl').exists():
        return artifacts
    candidates = sorted((p for p in artifacts.glob('*/progress.jsonl')),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].parent if candidates else artifacts


def build_handler(artifacts):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):        # keep the training console readable
            pass

        def _send(self, body, content_type):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith('/data.json'):
                payload = _read_progress(_resolve(artifacts) / 'progress.jsonl')
                self._send(json.dumps(payload).encode(), 'application/json')
            elif self.path in ('/', '/index.html'):
                self._send(DASHBOARD.read_bytes(), 'text/html; charset=utf-8')
            else:
                self.send_error(404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Live training dashboard for LesNet JEPA runs.")
    parser.add_argument('--artifacts', default='artifacts',
                        help="Run artifacts dir, or a parent containing several.")
    parser.add_argument('--port', type=int, default=6600)
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()

    resolved = _resolve(args.artifacts)
    url = f'http://127.0.0.1:{args.port}/'
    print(f'dashboard: {url}\nwatching:  {resolved / "progress.jsonl"}'
          f'{"" if (resolved / "progress.jsonl").exists() else "  (not created yet — will appear)"}',
          flush=True)
    if not args.no_open:
        webbrowser.open(url)
    HTTPServer(('127.0.0.1', args.port), build_handler(args.artifacts)).serve_forever()


if __name__ == '__main__':
    main()
