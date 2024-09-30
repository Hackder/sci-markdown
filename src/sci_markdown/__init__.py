import base64
import json
import os
import sys
import traceback
import uuid
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


chart_colors = [
    "100, 140, 255",
    "255, 100, 140",
    "50, 200, 50",
    "140, 255, 100",
    "255, 140, 100",
]


class Chartjs:
    def __init__(self, keys, chart_type="bar"):
        self.keys = keys
        self.chart_type = chart_type
        self.datasets = []

    def plot(self, data, label=None, char_type=None, color=None):
        dataset = {
            "data": list(data),
        }

        if label is not None:
            dataset["label"] = label

        if char_type is not None:
            dataset["type"] = char_type

        if color is not None:
            dataset["color"] = color

        self.datasets.append(dataset)

    def show(self):
        datasets = []
        for i, dataset in enumerate(self.datasets):
            color = chart_colors[i % len(chart_colors)]
            if "color" in dataset:
                color = dataset["color"]

            new_dataset = {
                "backgroundColor": f"rgba({color}, 0.5)",
                "borderColor": f"rgba({color}, 1)",
                "borderWidth": 1,
            }
            new_dataset.update(dataset)
            datasets.append(new_dataset)

        config = {
            "type": self.chart_type,
            "data": {
                "labels": list(self.keys),
                "datasets": datasets,
            },
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                    }
                },
            },
        }
        unique_id = str(uuid.uuid4())

        print(f"""
<canvas data-type="chartjs" id="{unique_id}">
<div data-chartid="{unique_id}">{json.dumps(config)}</div>
""")


def table(
    *_,
    header: list[Any] | None = None,
    left_header: list[Any] | None = None,
    rows: list[list[Any]] | None = None,
    cols: list[list[Any]] | None = None,
    corner: Any = None,
    precision=2,
):
    if cols is not None and rows is not None:
        raise ValueError("both cols and rows cannot be set")

    if cols is not None:
        table_rows = max(len(col) for col in cols)
        table_cols = len(cols)
    elif rows is not None:
        table_rows = len(rows)
        table_cols = max(len(row) for row in rows)
    else:
        raise ValueError("either cols or rows must be set")

    left_offset = 0

    table_rows += 1
    if header is not None:
        table_cols = max(table_cols, len(header))

    if left_header:
        table_rows = max(table_rows, len(left_header))

    if corner is not None or left_header is not None:
        table_cols += 1
        left_offset = 1

    cells: list[list[Any]] = [
        [None for _ in range(table_cols)] for _ in range(table_rows)
    ]

    if corner is not None:
        cells[0][0] = corner

    if header:
        for i, header in enumerate(header):
            cells[0][i + left_offset] = header

    if left_header:
        for i, header in enumerate(left_header):
            cells[i + 1][0] = header

    if cols is not None:
        for i, col in enumerate(cols):
            for j, cell in enumerate(col):
                cells[j + 1][i + left_offset] = cell

    if rows is not None:
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                cells[i + 1][j + left_offset] = cell

    def print_line(line: list[Any], data_line=True):
        def escape_none(val: tuple[int, Any]) -> str:
            i, value = val
            if value is None:
                return "<!-- -->"
            elif left_offset == 1 and i == 0 and data_line:
                return "**" + pstr(value, precision) + "**"
            else:
                return pstr(value, precision)

        print("|", " | ".join(map(escape_none, enumerate(line))), "|")

    print_line(cells[0])
    print_line(["-" for _ in cells[0]], data_line=False)
    for row in cells[1:]:
        print_line(row)


def img_plot(fig=None):
    if fig is None:
        fig = plt.gcf()

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
    buffer.seek(0)

    # Encode the BytesIO object to base64
    img_str = base64.b64encode(buffer.read()).decode("utf-8")

    buffer.close()
    print(f"![Plot](data:image/png;base64,{img_str})")


def __render(code: list[str | list[str]]) -> str:
    rendered_lines = []
    line_no = 0
    for line in code:
        if isinstance(line, list):
            code_string = "".join(line)
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            try:
                exec(code_string)
            except Exception as e:
                print("<pre class='python-error'>")
                cl, exc, tb = sys.exc_info()
                line_number = traceback.extract_tb(tb)[-1][1]
                file_name = traceback.extract_tb(tb)[-1][0]

                if len(file_name) <= 8:
                    line_number += line_no + 1

                file_name_text = ""
                if len(file_name) > 8:
                    file_name_text = f"in {file_name}"

                print(
                    f"<small>Exception on line: {line_number} {file_name_text}</small>"
                )
                print(e)
                print("</pre>")
            rendered_lines.append(mystdout.getvalue())
            sys.stdout = old_stdout
            line_no += len(line) + 2
        else:
            rendered_lines.append(line)
            line_no += 1

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


PORT = 8000


def main() -> int:
    compile_markdown()

    httpd = HTTPServer(("", PORT), SimpleHandler)
    try:
        print(f"Starting server on http://localhost:{PORT}")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, exiting")
        httpd.server_close()

    return 0
