import base64
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO, StringIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from markdown_it import MarkdownIt
from mdit_py_plugins import footnote


def table(header: list[Any], rows: list[list[Any]]) -> str:
    def format_line(line: list[Any]) -> str:
        return "|" + "|".join(map(str, line))

    header_line = format_line(header)
    separator_line = "|" + "|".join(["-" * len(str(x)) for x in header])
    body_lines = [format_line(row) for row in rows]

    data = "\n".join([header_line, separator_line] + body_lines)
    print(data)


def img_plot(fig=None):
    if fig is None:
        fig = plt.gcf()

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    # Encode the BytesIO object to base64
    img_str = base64.b64encode(buffer.read()).decode("utf-8")

    buffer.close()
    print(f"![Plot](data:image/png;base64,{img_str})")


def __render(code: list[str | list[str]]) -> str:
    rendered_lines = []
    for line in code:
        if isinstance(line, list):
            code_string = "".join(line)
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            exec(code_string)
            sys.stdout = old_stdout
            rendered_lines.append(mystdout.getvalue())
        else:
            rendered_lines.append(line)

    return "".join(rendered_lines)


md = (
    MarkdownIt(
        "commonmark",
        {
            "html": True,
            "breaks": False,
            "linkify": True,
            "typographer": True,
        },
    )
    .use(footnote.footnote_plugin)
    .enable("table")
)


latest_html = None
last_src = None


def compile_markdown():
    global latest_html, last_src
    filename = sys.argv[1]
    with open(filename, "r") as f:
        lines = f.readlines()

    if last_src == lines:
        return latest_html
    last_src = lines

    code = []

    it = iter(lines)
    for line in it:
        if line.startswith("```python exec"):
            line = next(it)
            code_block = []
            code_block.append(line)
            line = next(it)
            while not line.startswith("```"):
                code_block.append(line)
                line = next(it)
            code.append(code_block)
        else:
            code.append(line)

    new_content = __render(code)

    html = md.render(new_content)
    latest_html = html
    return html


class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_html
        # Set response code to 200 (OK)
        self.send_response(200)

        # Set the headers
        self.send_header("Content-type", "text/html")
        self.end_headers()

        if self.path == "/":
            with open(
                os.path.join(os.path.dirname(__file__), "default_page.html"), "r"
            ) as f:
                self.wfile.write(f.read().encode("utf-8"))
            return

        # Send the string as the response body
        html = str(compile_markdown())
        self.wfile.write(html.encode("utf-8"))


def main() -> int:
    compile_markdown()

    httpd = HTTPServer(("", 8000), SimpleHandler)
    httpd.serve_forever()

    return 0
