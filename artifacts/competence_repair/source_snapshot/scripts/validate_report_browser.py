"""Decode actual embedded report media in Chromium, including repaired-model clips."""

import base64
import json
import socket
import subprocess
import tempfile
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from websockets.sync.client import connect


class Media(HTMLParser):
    def __init__(self):
        super().__init__()
        self.videos, self.images = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "video":
            self.videos.append(attrs["src"])
        if attrs.get("data-policy"):
            self.videos.append(attrs["data-policy"])
        if tag == "img" and attrs.get("src", "").startswith("data:image/svg"):
            self.images.append(attrs["src"])


root = Path(__file__).resolve().parents[1]
out = root / "artifacts/training_results"
m = Media()
m.feed((out / "index.html").read_text())
analysis = json.loads((out / "latest_analysis.json").read_text())
repair_count = analysis.get("repair", {}).get("media_examples", 0)
selected = m.videos[:2] + m.videos[-max(2, 2 * repair_count) :]
script = """
const sources=SOURCES; const results=[];
function finish(){if(results.length===sources.length){document.getElementById('result').textContent=JSON.stringify({videos:results,plots:[...document.images].map(x=>({decoded:x.complete&&x.naturalWidth>0,width:x.naturalWidth})),background:getComputedStyle(document.body).backgroundColor});}}
sources.forEach((src,index)=>{const v=document.createElement('video');v.muted=true;v.controls=true;v.preload='auto';v.onloadeddata=()=>{v.currentTime=Math.min(.4,v.duration/2);};v.onseeked=()=>{results.push({index,decoded:true,width:v.videoWidth,height:v.videoHeight,time:v.currentTime,duration:v.duration});v.onseeked=null;finish();};v.onerror=()=>{results.push({index,decoded:false,error:v.error.code});finish();};v.src=src;document.body.appendChild(v);});
""".replace("SOURCES", json.dumps(selected))
with tempfile.TemporaryDirectory(prefix="report-browser-") as folder:
    path = Path(folder) / "probe.html"
    path.write_text(
        '<!doctype html><style>body{background:#fff}img,video{width:240px}</style><pre id="result">PENDING</pre>'
        + "".join(f'<img src="{x}">' for x in m.images)
        + f"<script>{script}</script>"
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (Path(folder) / "chromium.log").open("w") as log:
        browser = subprocess.Popen(
            [
                "chromium-browser",
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-zygote",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={folder}/profile",
                "--autoplay-policy=no-user-gesture-required",
                "about:blank",
            ],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{port}/json/new?{path.as_uri()}", method="PUT"
                    )
                    target = json.load(urllib.request.urlopen(request, timeout=2))
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
            with connect(target["webSocketDebuggerUrl"], max_size=100_000_000) as ws:
                request_id = 0

                def call(method, params):
                    global request_id
                    request_id += 1
                    ws.send(json.dumps(dict(id=request_id, method=method, params=params)))
                    while True:
                        message = json.loads(ws.recv(timeout=20))
                        if message.get("id") == request_id:
                            assert "error" not in message, message
                            return message["result"]

                deadline = time.monotonic() + 40
                while True:
                    reply = call(
                        "Runtime.evaluate",
                        {
                            "expression": "document.getElementById('result')?.textContent",
                            "returnByValue": True,
                        },
                    )
                    value = reply["result"].get("value")
                    if value and value != "PENDING":
                        result = json.loads(value)
                        break
                    assert time.monotonic() < deadline, (
                        "Embedded media did not finish browser decoding"
                    )
                    time.sleep(0.25)
                assert len(result["videos"]) == len(selected)
                assert all(
                    v["decoded"] and v["width"] > 0 and v["time"] > 0 for v in result["videos"]
                )
                assert all(p["decoded"] for p in result["plots"])
                assert result["background"] == "rgb(255, 255, 255)"
                call("Page.navigate", {"url": (out / "index.html").as_uri()})
                time.sleep(2)
                call(
                    "Emulation.setDeviceMetricsOverride",
                    {"width": 1200, "height": 1600, "deviceScaleFactor": 1, "mobile": False},
                )
                screenshot = call("Page.captureScreenshot", {"format": "png"})
                Path("/tmp/latest-repair-report.png").write_bytes(
                    base64.b64decode(screenshot["data"])
                )
                (out / "browser_playback_validation.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print(
                    f"Chromium decoded and sought {len(selected)} embedded videos; rendered {len(result['plots'])} plots on white background."
                )
        finally:
            browser.terminate()
            browser.wait(timeout=10)
