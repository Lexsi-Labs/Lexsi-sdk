import math


def in_notebook() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _clean(name: str) -> str:
    return name.replace("system/", "")


class LiveMetricsPlotter:

    def __init__(self, title: str, x_title: str, mode: str = "lines+markers", cols: int = 3):
        self.title = title
        self.x_title = x_title
        self.mode = mode
        self.cols = cols
        self._fig = None
        self._out = None
        self._trace_index = {}  # metric name -> trace index within fig.data

    def update(self, metrics: dict) -> None:
        active = {name: points for name, points in (metrics or {}).items() if points}
        if not active:
            return

        names = sorted(active.keys())

        if self._fig is None or any(name not in self._trace_index for name in names):
            self._build(names)

        for name in names:
            idx = self._trace_index.get(name)
            if idx is None:
                continue
            points = active[name]
            self._fig.data[idx].x = [p[0] for p in points]
            self._fig.data[idx].y = [p[1] for p in points]

    def _build(self, names) -> None:
        """Create (or recreate) the subplot grid for ``names`` and display it."""
        import ipywidgets as widgets
        import plotly.graph_objects as go
        from IPython.display import display
        from plotly.subplots import make_subplots

        cols = min(self.cols, len(names))
        rows = math.ceil(len(names) / cols)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[_clean(name) for name in names],
            horizontal_spacing=min(0.2, 0.4 / cols),
            vertical_spacing=min(0.25, 0.6 / rows),
        )
        figw = go.FigureWidget(fig)

        for annotation in figw.layout.annotations:
            annotation.font.size = 11

        self._trace_index = {}
        for i, name in enumerate(names):
            row = i // cols + 1
            col = i % cols + 1
            figw.add_scatter(x=[], y=[], mode=self.mode, name=_clean(name), row=row, col=col)
            self._trace_index[name] = i

        figw.update_layout(
            height=260 * rows,
            showlegend=False,
            title_text=self.title,
            margin=dict(t=70, l=40, r=20, b=40),
        )
        figw.update_xaxes(title_text=self.x_title)
        self._fig = figw

        if self._out is None:
            self._out = widgets.Output()
            display(self._out)
        self._out.clear_output(wait=True)
        with self._out:
            display(self._fig)
