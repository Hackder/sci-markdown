import base64
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO, StringIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from markdown_it import MarkdownIt
from mdit_py_plugins import footnote


def pstr(val, precision=2) -> str:
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    else:
        return str(val)


def pprint(*args, precision=2, **kwargs):
    str_args = []
    for arg in args:
        if isinstance(arg, float):
            str_args.append(f"{arg:.{precision}f}")
        else:
            str_args.append(str(arg))
    print(*str_args, **kwargs)


def ctable(header: list[Any], rows: list[list[Any]], precision=2):
    def format_line(line: list[Any]) -> str:
        return "|" + "|".join(map(lambda x: pstr(x, precision), line))

    header_line = format_line(header)
    separator_line = "|" + "|".join(["-" for _ in header])
    body_lines = [format_line(row) for row in rows]

    data = "\n".join([header_line, separator_line] + body_lines)
    pprint(data)


def rtable(header: list[Any], rows: list[list[Any]], precision=2):
    def format_line(header: Any, line: list[Any]) -> str:
        return (
            "| **"
            + pstr(header, precision)
            + "**"
            + "|"
            + "|".join(map(lambda x: pstr(x, precision), line))
        )

    def header_value(index: int) -> Any:
        if index < len(header):
            return header[index]
        else:
            return "<!-- -->"

    header_line = format_line("<!-- -->", ["<!-- -->" for _ in header])
    separator_line = "|-|" + "|".join(["-" for _ in header])
    body_lines = [format_line(header_value(i), row) for i, row in enumerate(rows)]

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
            try:
                exec(code_string)
            except Exception as e:
                print("<pre class='python-error'>")
                if e.__traceback__:
                    print(f"<small>Error on line {e.__traceback__.tb_lineno}:</small>")
                print(e)
                print("</pre>")
            rendered_lines.append(mystdout.getvalue())
            sys.stdout = old_stdout
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
    try:
        print("Starting server on port 8000")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, exiting")
        httpd.server_close()

    return 0
