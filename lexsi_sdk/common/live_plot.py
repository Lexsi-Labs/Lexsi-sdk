import math


def in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and hasattr(ip, "kernel")
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
        self._display_id = f"lexsi-live-metrics-{id(self)}"
        self._displayed = False
        self._trace_index = {}
        self._trace_len = {}

    def update(self, metrics: dict) -> None:
        """Push the latest metric series into the figure, re-rendering if changed.

        :param metrics: mapping of ``{metric_name: [[x, y], ...]}``.
        """
        active = {name: points for name, points in (metrics or {}).items() if points}
        if not active:
            return

        names = sorted(active.keys())
        rebuilt = False
        if self._fig is None or any(name not in self._trace_index for name in names):
            self._build(names)
            rebuilt = True

        changed = rebuilt
        for name in names:
            idx = self._trace_index.get(name)
            if idx is None:
                continue
            points = active[name]
            if not rebuilt and self._trace_len.get(name) == len(points):
                continue
            self._trace_len[name] = len(points)
            self._fig.data[idx].x = [p[0] for p in points]
            self._fig.data[idx].y = [p[1] for p in points]
            changed = True

        if changed:
            self._render()

    def _build(self, names) -> None:
        """Create (or recreate) the subplot grid for ``names``."""
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

        self._trace_index = {}
        self._trace_len = {}
        for i, name in enumerate(names):
            row = i // cols + 1
            col = i % cols + 1
            fig.add_scatter(x=[], y=[], mode=self.mode, name=_clean(name), row=row, col=col)
            self._trace_index[name] = i

        for annotation in fig.layout.annotations:
            annotation.font.size = 11

        fig.update_layout(
            height=260 * rows,
            showlegend=False,
            title_text=self.title,
            margin=dict(t=70, l=40, r=20, b=40),
        )
        fig.update_xaxes(title_text=self.x_title)
        self._fig = fig

    def _render(self) -> None:
        import json

        from IPython.display import display, update_display

        bundle = {
            "application/vnd.plotly.v1+json": json.loads(self._fig.to_json()),
            "text/html": self._fig.to_html(full_html=False, include_plotlyjs="cdn"),
        }
        if not self._displayed:
            display(bundle, raw=True, display_id=self._display_id)
            self._displayed = True
        else:
            update_display(bundle, raw=True, display_id=self._display_id)
