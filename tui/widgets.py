from textual.widgets import Static, DataTable, ProgressBar
from textual.containers import Container, Vertical, Horizontal
from textual.reactive import reactive
from rich.text import Text
from rich.console import RenderableType
import numpy as np


class TranscriptDisplay(Static):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lines: list[str] = []

    def compose(self):
        yield Static(id="transcript-content")

    def clear(self) -> None:
        self._lines.clear()
        self._update_display()

    def add_response(self, model: str, content: str, round_num: int, response_time_s: float | None = None) -> None:
        time_str = ""
        if response_time_s is not None:
            if response_time_s >= 1:
                time_str = f" — {response_time_s:.1f}s"
            else:
                time_str = f" — {response_time_s*1000:.0f}ms"
        self._lines.append(f"[bold]● {model} (Round {round_num}){time_str}[/bold]\n{content}")
        self._update_display()

    def add_summary(self, summary: str, round_num: int) -> None:
        self._lines.append(f"[bold cyan]📝 Summary (Round {round_num})[/bold cyan]\n{summary}")
        self._update_display()

    def add_system_message(self, message: str) -> None:
        self._lines.append(f"[dim italic]{message}[/dim italic]")
        self._update_display()

    def _update_display(self) -> None:
        content_widget = self.query_one("#transcript-content", Static)
        if not self._lines:
            content_widget.update("")
        else:
            # Join with double newlines and use from_markup for basic styling
            from rich.console import Console

            combined = "\n\n".join(self._lines)
            content_widget.update(combined)


class SimilarityMatrix(Static):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._matrix: np.ndarray | None = None
        self._model_names: list[str] = []

    def compose(self):
        yield Static(id="matrix-content")

    async def update_from_orchestrator(self, orchestrator, round_num: int) -> None:
        if not orchestrator or not orchestrator.session:
            return

        round_responses = orchestrator.session.get_round_responses(round_num)
        if len(round_responses) < 2:
            return

        texts = [r.content for r in round_responses]
        self._model_names = [r.model for r in round_responses]

        similarity_result = await orchestrator.similarity_engine.calculate_similarity_matrix(
            texts, self._model_names
        )

        self._matrix = similarity_result.matrix
        self._update_display()

    def set_matrix(self, matrix: np.ndarray, model_names: list[str]) -> None:
        self._matrix = matrix
        self._model_names = model_names
        self._update_display()

    def _update_display(self) -> None:
        if self._matrix is None:
            return

        content_widget = self.query_one("#matrix-content", Static)

        lines = ["Similarity Matrix:"]
        n = len(self._model_names)

        header = "       " + "  ".join(f"{name[:6]:>6}" for name in self._model_names)
        lines.append(header)

        for i in range(n):
            row_values = []
            for j in range(n):
                if i == j:
                    row_values.append("   -  ")
                else:
                    sim = self._matrix[i, j]
                    row_values.append(f"{sim:.2f}")
            lines.append(f"{self._model_names[i][:6]:>6}  {'  '.join(row_values)}")

        content_widget.update("\n".join(lines))


class StatusPanel(Static):
    current_round: reactive[int] = reactive(1)
    max_rounds: reactive[int] = reactive(10)
    current_model: reactive[str] = reactive("")
    consensus_percentage: reactive[float] = reactive(0.0)
    is_running: reactive[bool] = reactive(False)
    is_paused: reactive[bool] = reactive(False)

    def compose(self):
        yield Static(id="status-content")

    def update_state(self, state) -> None:
        self.current_round = state.current_round
        self.current_model = (
            state.model_order[state.current_model_index] if state.model_order else ""
        )
        self.is_running = state.is_running
        self.is_paused = state.is_paused

        if state.consensus_result:
            self.consensus_percentage = state.consensus_result.percentage

        self._update_display()

    def _update_display(self) -> None:
        content_widget = self.query_one("#status-content", Static)

        status_icon = "⏸" if self.is_paused else "▶" if self.is_running else "⏹"
        status_color = "yellow" if self.is_paused else "green" if self.is_running else "red"

        progress = self.consensus_percentage / 100

        lines = [
            f"[{status_color}]{status_icon}[/{status_color}] Status: {'Running' if self.is_running else 'Paused' if self.is_paused else 'Stopped'}",
            "",
            f"Round: {self.current_round}/{self.max_rounds}",
            f"Current: {self.current_model}",
            "",
            f"Consensus: {self.consensus_percentage:.0f}%",
        ]

        content_widget.update("\n".join(lines))


class ModelSelector(Static):
    def __init__(self, models: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = models
        self.selected: list[str] = []

    def compose(self):
        yield Static("Select Models:", id="selector-label")
        for model in self.models:
            yield Static(f"☐ {model}", classes="model-option", id=f"model-{model}")
