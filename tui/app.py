from textual.app import App, ComposeResult
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Header,
    Footer,
    Static,
    Button,
    Input,
    Label,
    ProgressBar,
    DataTable,
    RichLog,
)
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.binding import Binding
from textual import work
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
import asyncio

from core import Config, DiscussionOrchestrator, OllamaClient
from core.discussion import DiscussionState
from storage import Session, SessionManager, Exporter
from tui.widgets import (
    TranscriptDisplay,
    SimilarityMatrix,
    StatusPanel,
    ModelSelector,
)
from tui.screens import (
    PromptScreen,
    ConfigScreen,
    SessionListScreen,
    ExportScreen,
)


class RoundtableApp(App):
    CSS_PATH = "styles.css"

    BINDINGS = [
        Binding("ctrl+p", "start_discussion", "Start", priority=True),
        Binding("ctrl+s", "save_session", "Save"),
        Binding("ctrl+e", "export", "Export"),
        Binding("ctrl+o", "open_session", "Open"),
        Binding("ctrl+c", "quit", "Quit"),
        Binding("space", "toggle_pause", "Pause/Resume"),
        Binding("ctrl+q", "stop_discussion", "Stop"),
    ]

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path
        self.config: Config | None = None
        self.orchestrator: DiscussionOrchestrator | None = None
        self.session: Session | None = None
        self.session_manager: SessionManager | None = None
        self.similarity_matrix: list[list[float]] = []
        self.model_names: list[str] = []
        self.discussion_running = False
        self.discussion_paused = False
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                TranscriptDisplay(id="transcript"),
                id="main-panel",
            ),
            Vertical(
                StatusPanel(id="status-panel"),
                SimilarityMatrix(id="similarity-matrix"),
                Container(
                    Button("⏸ Pause", id="pause-btn"),
                    Button("⏭ Skip Round", id="skip-btn"),
                    Button("🛑 Stop", id="stop-btn", variant="error"),
                    Button("💾 Export", id="export-btn", variant="primary"),
                    id="controls",
                ),
                id="side-panel",
            ),
            id="content",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "LLM Roundtable Discussion"
        self.sub_title = "Loading..."
        self._load_config()

    def _load_config(self) -> None:
        try:
            self.config = Config.load(self.config_path)
            self.session_manager = SessionManager(self.config.storage.sessions_dir)
            self.sub_title = f"Configured with {len(self.config.models)} models"
            self._show_prompt_screen()
        except Exception as e:
            self.notify(f"Error loading config: {e}", severity="error")
            self.exit(1)

    def _show_prompt_screen(self) -> None:
        self.push_screen(PromptScreen(config=self.config))

    def _show_session_list(self) -> None:
        async def load_sessions():
            sessions = await self.session_manager.list_sessions()
            return sessions

        async def on_select(session_id: str):
            await self._load_session(session_id)

        self.push_screen(SessionListScreen(load_sessions, on_select))

    async def _load_session(self, session_id: str) -> None:
        session = await self.session_manager.load(session_id)
        if session:
            self.session = session
            self._initialize_orchestrator()
            self._update_transcript_from_session()
            self.notify(f"Loaded session: {session.prompt[:50]}...")
        else:
            self.notify("Session not found", severity="error")

    def _initialize_orchestrator(self) -> None:
        if self.config and self.session:
            self.orchestrator = DiscussionOrchestrator(
                config=self.config,
                session=self.session,
                progress_callback=self._on_progress_update,
            )

    def _update_transcript_from_session(self) -> None:
        if not self.session:
            return

        transcript = self.query_one("#transcript", TranscriptDisplay)
        transcript.clear()

        for response in self.session.responses:
            transcript.add_response(
                model=response.model,
                content=response.content,
                round_num=response.round,
            )

        for summary in self.session.summaries:
            transcript.add_summary(summary.summary, summary.round)

    async def _on_progress_update(self, state: DiscussionState) -> None:
        self.model_names = state.model_order

        status_panel = self.query_one("#status-panel", StatusPanel)
        status_panel.update_state(state)

        similarity_matrix = self.query_one("#similarity-matrix", SimilarityMatrix)
        if state.consensus_result and hasattr(state.consensus_result, "details"):
            await similarity_matrix.update_from_orchestrator(self.orchestrator, state.current_round)

        transcript = self.query_one("#transcript", TranscriptDisplay)
        transcript.scroll_end()

        self.discussion_running = state.is_running
        self.discussion_paused = state.is_paused

        pause_btn = self.query_one("#pause-btn", Button)
        pause_btn.label = "▶ Resume" if state.is_paused else "⏸ Pause"

    @work(exclusive=True)
    async def _run_discussion(self, prompt: str) -> None:
        if not self.config:
            return

        self.session = Session(prompt=prompt, config=self.config.model_dump())
        self._initialize_orchestrator()

        transcript = self.query_one("#transcript", TranscriptDisplay)
        transcript.clear()
        transcript.add_system_message(f"Starting discussion with prompt:\n\n{prompt}")

        try:
            await self.orchestrator.run()
            self.notify("Discussion completed!", severity="information")
            self._update_transcript_from_session()

            if self.session.consensus_reached:
                self.notify(
                    f"Consensus reached in round {self.session.consensus_round}!",
                    severity="information",
                )
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
        finally:
            if self.orchestrator:
                await self.orchestrator.cleanup()
            self.discussion_running = False

    def action_start_discussion(self) -> None:
        if self.discussion_running:
            self.notify("Discussion already running", severity="warning")
            return
        self._show_prompt_screen()

    def action_toggle_pause(self) -> None:
        if not self.discussion_running:
            return

        if self.orchestrator:
            if self.discussion_paused:
                asyncio.create_task(self.orchestrator.resume())
                self.notify("Discussion resumed")
            else:
                asyncio.create_task(self.orchestrator.pause())
                self.notify("Discussion paused")

    def action_stop_discussion(self) -> None:
        if not self.discussion_running:
            return

        if self.orchestrator:
            asyncio.create_task(self.orchestrator.stop())
            self.notify("Stopping discussion...")

    def action_save_session(self) -> None:
        if self.session and self.session_manager:

            async def save():
                path = await self.session_manager.save(self.session)
                self.notify(f"Saved to {path}")

            asyncio.create_task(save())
        else:
            self.notify("No active session to save", severity="warning")

    def action_export(self) -> None:
        if not self.session:
            self.notify("No session to export", severity="warning")
            return

        self.push_screen(ExportScreen(self.session, self.config.storage.export_format))

    def action_open_session(self) -> None:
        if self.discussion_running:
            self.notify("Cannot open session while discussion is running", severity="warning")
            return
        self._show_session_list()

    def on_prompt_screen_submitted(self, message: PromptScreen.Submitted) -> None:
        self.pop_screen()
        self._run_discussion(message.prompt)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "pause-btn":
            self.action_toggle_pause()
        elif button_id == "stop-btn":
            self.action_stop_discussion()
        elif button_id == "export-btn":
            self.action_export()
        elif button_id == "skip-btn":
            self.notify("Skip round not yet implemented", severity="warning")


def run_tui(config_path: str | None = None) -> None:
    app = RoundtableApp(config_path=config_path)
    app.run()
