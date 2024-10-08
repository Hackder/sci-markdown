import asyncio
import base64
import functools
import json
import os
import sys
import traceback
import uuid
from io import BytesIO, StringIO
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from markdown_it import MarkdownIt
from mdit_py_plugins import footnote


def simplify_fraction(numer: int, denom: int):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    common_divisor = gcd(numer, denom)
    return numer // common_divisor, denom // common_divisor


def frac(x: float, precision=2) -> str:
    for numer in range(-256, 256):
        if numer == 0:
            continue
        for denom in range(1, 256):
            if abs(x - numer / denom) < 1e-6:
                n, d = simplify_fraction(numer, denom)
                if n < 0:
                    return rf"-$\frac{{{-n}}}{{{d}}}$"
                else:
                    return rf"$\frac{{{n}}}{{{d}}}$"

    num_str = f"{x:.{precision}f}".rstrip("0").rstrip(".")
    return f"${num_str}$"


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

    def plot(self, data, label=None, chart_type=None, color=None, stepped=None):
        dataset = {
            "data": list(data),
        }

        if label is not None:
            dataset["label"] = label

        if chart_type is not None:
            dataset["type"] = chart_type

        if color is not None:
            dataset["color"] = color

        if stepped is not None:
            dataset["stepped"] = stepped

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
                    },
                },
            },
        }
        unique_id = str(uuid.uuid4())

        print(f"""<canvas data-type="chartjs" id="{unique_id}">
<div data-chartid="{unique_id}">{json.dumps(config)}</div>
</canvas>
""")


class Graph:
    def __init__(self, xrange=[-1, 1], yrange=[-1, 1], resolution=100):
        self.xrange = xrange
        self.yrange = yrange
        self.resolution = resolution
        self.fig, self.ax = plt.subplots()
        self.fig.set_figwidth(10)
        self.fig.set_figheight(7)
        self.ax.set_xlim(xrange)
        self.ax.set_ylim(yrange)

    def plot(self, function):
        keys = np.linspace(self.xrange[0], self.xrange[1], self.resolution)
        values = function(keys)
        if isinstance(values, (int, float)):
            values = np.linspace(values, values, self.resolution)

        self.ax.plot(keys, values)

    def plot_between(
        self,
        fu1: Callable[[float], float | int],
        fn2: Callable[[float], float | int],
        compare: str | None = None,
        **kwargs,
    ):
        custom_kwargs = {
            "alpha": 0.5,
            "interpolate": True,
        }
        custom_kwargs.update(kwargs)

        values = np.linspace(self.xrange[0], self.xrange[1], 100)
        self.ax.fill_between(values, fu1(values), fn2(values), **custom_kwargs)

    def show(self):
        plt.grid(True)
        img_plot(self.fig)


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

    def table_pstr(val: Any, *args, **kwargs) -> str:
        return pstr(val, *args, **kwargs).replace("|", "&#124;")

    def print_line(line: list[Any], data_line=True):
        def escape_none(val: tuple[int, Any]) -> str:
            i, value = val
            if value is None:
                return "<!-- -->"
            elif left_offset == 1 and i == 0 and data_line:
                return "**" + table_pstr(value, precision) + "**"
            else:
                return table_pstr(value, precision)

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


def mermaid_plugin(md: MarkdownIt):
    default_fence = md.renderer.rules.get("fence", None)

    def fence_with_mermaid(tokens, idx, options, env):
        token = tokens[idx]
        info = token.info.strip()
        language = info.split()[0] if info else ""

        if language == "mermaid":
            # Render as a <pre> with class="mermaid"
            return f'<pre class="mermaid">{md.utils.escapeHtml(token.content)}</pre>\n'
        else:
            # Fallback to the default fence rule for other languages
            if default_fence:
                return default_fence(tokens, idx, options, env)
            else:
                return f"<pre>{md.utils.escapeHtml(token.content)}</pre>\n"

    # Override the fence rule
    md.renderer.rules["fence"] = fence_with_mermaid


class Markdown:
    def __init__(self, content: str):
        self.content = content


class Code:
    def __init__(self, content: str):
        self.content = content


Token = Markdown | Code


def what_line(until_now: str) -> int:
    return until_now.count("\n") + 1


@functools.lru_cache(maxsize=32)
def parse_source(src: str) -> list[Token]:
    tokens = []
    pos = 0
    read_pos = 0
    while read_pos < len(src):
        if src[read_pos : read_pos + 3] == "py`":
            tokens.append(Markdown(src[pos:read_pos] + " "))
            pos = read_pos + 3
            read_pos += 3
            while src[read_pos] != "`":
                if src[read_pos] == "\n":
                    raise ValueError(
                        f"Unclosed inline code block, on line {what_line(src[:read_pos])}"
                    )
                read_pos += 1
                if read_pos >= len(src):
                    raise ValueError(
                        f"Unclosed inline code block. on line {what_line(src[:pos])}"
                    )
            tokens.append(Code(src[pos:read_pos]))
            pos = read_pos + 1

        if src[read_pos : read_pos + 15] == "```python exec\n":
            tokens.append(Markdown(src[pos:read_pos] + "\n"))
            pos = read_pos + 15
            read_pos += 15
            while src[read_pos : read_pos + 4] != "\n```":
                read_pos += 1
                if read_pos >= len(src):
                    raise ValueError(
                        f"Unclosed code block. Opened on line {what_line(src[:pos])}"
                    )
            tokens.append(Code(src[pos:read_pos]))
            pos = read_pos + 4

        read_pos += 1

    if pos < len(src):
        tokens.append(Markdown(src[pos:]))

    return tokens


def __render(code: list[Token]) -> str:
    rendered_parts = []
    line_no = 0
    global_ctx = {
        "frac": frac,
        "pprint": pprint,
        "pstr": pstr,
        "table": table,
        "Chartjs": Chartjs,
        "Graph": Graph,
        "img_plot": img_plot,
    }
    for token in code:
        if isinstance(token, Code):
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            try:
                exec(token.content, global_ctx)
            except Exception as e:
                cl, exc, tb = sys.exc_info()
                line_number = traceback.extract_tb(tb)[-1][1]
                file_name = traceback.extract_tb(tb)[-1][0]

                if len(file_name) <= 8:
                    line_number += line_no + 1

                file_name_text = ""
                if len(file_name) > 8:
                    file_name_text = f"in {file_name}"

                print('<pre class="python-error">')
                print(
                    f"<small>Exception on line: {line_number} {file_name_text}</small>"
                )
                print(e)
                print("</pre>")
            rendered_parts.append(mystdout.getvalue())
            sys.stdout = old_stdout
            line_no += len(token.content.split("\n")) + 2
        else:
            rendered_parts.append(token.content)
            line_no += 1

    plt.close("all")
    return "".join(rendered_parts)


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
    .use(mermaid_plugin)
    .enable("table")
)


def read_source():
    filename = sys.argv[1]
    with open(filename, "r") as f:
        content = f.read()

    return content


def compile_markdown(content: str):
    tokens = parse_source(content)
    new_content = __render(tokens)
    html = md.render(new_content)
    return html


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(os.path.dirname(__file__), "default_page.html"), "r") as f:
        return f.read()


sockets = []


@app.websocket("/live-update")
async def ws_live_update(websocket: WebSocket):
    global sockets
    await websocket.accept()
    sockets.append(websocket)
    source = None
    try:
        while True:
            await asyncio.sleep(0.5)
            await websocket.send_json({"type": "heartbeat"})

            new_source = read_source()
            if new_source == source:
                continue

            source = new_source
            try:
                compiled = compile_markdown(new_source)
                await websocket.send_json({"type": "update", "data": compiled})
            except ValueError as e:
                await websocket.send_json({"type": "error", "data": str(e)})

    except WebSocketDisconnect:
        sockets.remove(websocket)


def main():
    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    main()
