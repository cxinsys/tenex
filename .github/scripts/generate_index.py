"""Generate a PEP 503 simple index from GitHub Release wheels, split by CUDA line.

Each CUDA line (cpu, cu126, cu128, cu132, ...) gets its own sub-index, so that

    pip install tnx --extra-index-url https://cxinsys.github.io/tenex/whl/cu132/

resolves to exactly the wheels built for that line. A single flat index does not
work here: the CUDA wheels carry local version labels (0.1.0+pt213cu132), and pip
would pick the highest-sorting label for the platform regardless of which torch is
installed, which can mismatch the installed torch and crash on import.

The files are written under `_site/`, which the workflow deploys to the gh-pages
`whl/` directory (destination_dir: whl). So the paths here are relative to `whl/`
and must NOT include a `whl/` prefix of their own.
"""
import os
import re
import pathlib

import requests

OWNER = "cxinsys"
REPO = "tenex"
PACKAGE = "tnx"


def cuda_line(filename):
    """Return the CUDA line for a wheel name.

    'tnx-0.1.0+pt213cu132-cp312-...' -> 'cu132'
    'tnx-0.1.0-py3-none-any.whl'     -> 'cpu'
    """
    m = re.search(r"\+[0-9a-z]*?(cu\d+)", filename)
    return m.group(1) if m else "cpu"


def _simple_page(links):
    return "<!DOCTYPE html>\n<html><body>\n" + "\n".join(links) + "\n</body></html>\n"


def main():
    api = f"https://api.github.com/repos/{OWNER}/{REPO}/releases"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    releases = []
    page = 1
    while True:
        resp = requests.get(
            api, headers=headers, params={"page": page, "per_page": 100}
        ).json()
        if isinstance(resp, dict) and "message" in resp:
            print(f"API error: {resp['message']}")
            break
        if not resp:
            break
        releases.extend(resp)
        page += 1

    # Bucket each wheel by CUDA line.
    buckets = {}
    for release in releases:
        for asset in release.get("assets", []):
            name = asset["name"]
            if name.endswith(".whl"):
                buckets.setdefault(cuda_line(name), []).append(
                    (name, asset["browser_download_url"])
                )

    site = pathlib.Path("_site")
    site.mkdir(parents=True, exist_ok=True)

    # One PEP 503 sub-index per line, deployed to whl/<line>/tnx/index.html
    for line, wheels in buckets.items():
        pkg_dir = site / line / PACKAGE
        pkg_dir.mkdir(parents=True, exist_ok=True)
        links = [
            f'    <a href="{url}" data-requires-python="&gt;=3.10">{name}</a><br/>'
            for name, url in sorted(wheels)
        ]
        (pkg_dir / "index.html").write_text(_simple_page(links))
        (site / line / "index.html").write_text(
            _simple_page([f'    <a href="{PACKAGE}/">{PACKAGE}</a><br/>'])
        )

    # Human-readable landing page (served at whl/) listing the available lines.
    line_links = "\n".join(
        f'    <a href="{line}/">{line}</a> ({len(w)} wheels)<br/>'
        for line, w in sorted(buckets.items())
    )
    (site / "index.html").write_text(
        "<!DOCTYPE html>\n<html><body>\n"
        "<p>TENEX wheel index. Install torch for your CUDA line first, then use the "
        "matching sub-index, for example<br/>\n"
        "<code>pip install tnx --extra-index-url "
        "https://cxinsys.github.io/tenex/whl/cu132/</code></p>\n"
        + line_links
        + "\n</body></html>\n"
    )

    total = sum(len(w) for w in buckets.values())
    print(f"Generated index: {total} wheels across lines {sorted(buckets)}")


if __name__ == "__main__":
    main()
