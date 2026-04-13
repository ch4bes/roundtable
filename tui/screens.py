from textual.screen import ModalScreen
from textual.widgets import (
    Static,
    Button,
    Input,
    Label,
    DataTable,
    TextArea,
)
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.message import Message
import asyncio
from rich.text import Text
from rich.panel import Panel


class PromptScreen(ModalScreen):
    class Submitted(Message):
        def __init__(self, prompt: str):
            super().__init__()
            self.prompt = prompt

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def compose(self):
        yield Vertical(
            Static("Enter Discussion Prompt", classes="modal-title"),
            TextArea(
                self.config.default_prompt if self.config.default_prompt else "",
                id="prompt-input",
            ),
            Horizontal(
                Button("Cancel", id="cancel", variant="default"),
                Button("Start Discussion", id="start", variant="primary"),
            ),
            classes="modal-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            textarea = self.query_one("#prompt-input", TextArea)
            prompt = textarea.text.strip()
            if prompt:
                self.post_message(self.Submitted(prompt))
            else:
                self.notify("Please enter a prompt", severity="warning")
        elif event.button.id == "cancel":
            self.app.pop_screen()


class ConfigScreen(ModalScreen):
    def compose(self):
        yield Vertical(
            Static("Configuration (JSON)", classes="modal-title"),
            TextArea(id="config-input", language="json"),
            Horizontal(
                Button("Cancel", id="cancel", variant="default"),
                Button("Save", id="save", variant="primary"),
            ),
            classes="modal-container",
        )


class SessionListScreen(ModalScreen):
    def __init__(self, load_sessions_fn, on_select_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_sessions_fn = load_sessions_fn
        self._on_select_fn = on_select_fn

    def compose(self):
        yield Vertical(
            Static("Saved Sessions", classes="modal-title"),
            DataTable(id="sessions-table"),
            Horizontal(
                Button("Cancel", id="cancel", variant="default"),
                Button("Load", id="load", variant="primary"),
                Button("Delete", id="delete", variant="error"),
            ),
            classes="modal-container",
        )

    def on_mount(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        table.add_column("ID", width=40)
        table.add_column("Prompt", width=60)
        table.add_column("Status", width=12)
        table.add_column("Rounds", width=8)
        table.add_column("Created", width=24)

        import asyncio

        async def load():
            sessions = await self._load_sessions_fn()
            for session in sessions:
                table.add_row(
                    session["id"][:36],
                    session["prompt"][:58] + "..."
                    if len(session["prompt"]) > 60
                    else session["prompt"],
                    session["status"],
                    str(session.get("completed_rounds", 0)),
                    session["created_at"][:19],
                    key=session["id"],
                )

        asyncio.create_task(load())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        table = self.query_one("#sessions-table", DataTable)

        if event.button.id == "load":
            if table.cursor_row is not None:
                row_key = table.get_row_at(table.cursor_row).key
                if row_key:
                    asyncio.create_task(self._on_select_fn(row_key))
                    self.app.pop_screen()
        elif event.button.id == "delete":
            self.notify("Delete not implemented", severity="warning")
        elif event.button.id == "cancel":
            self.app.pop_screen()


class ExportScreen(ModalScreen):
    def __init__(self, session, default_format, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = session
        self.default_format = default_format

    def compose(self):
        yield Vertical(
            Static("Export Discussion", classes="modal-title"),
            Static(f"Session: {self.session.id[:36]}", id="session-info"),
            Static(f"Prompt: {self.session.prompt[:80]}...", id="prompt-preview"),
            Horizontal(
                Button("Markdown (.md)", id="export-md", variant="primary"),
                Button("JSON (.json)", id="export-json", variant="primary"),
            ),
            Button("Cancel", id="cancel", variant="default"),
            classes="modal-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id in ["export-md", "export-json"]:
            format = "md" if event.button.id == "export-md" else "json"
            import asyncio
            from pathlib import Path

            async def do_export():
                from storage import Exporter

                filename = f"discussion_{self.session.id[:8]}.{format}"
                path = (
                    Path(
                        self.session.config_snapshot.get("storage", {}).get(
                            "sessions_dir", "./sessions"
                        )
                    )
                    / filename
                )
                await Exporter.export(self.session, path, format)
                self.notify(f"Exported to {path}", severity="information")
                self.app.pop_screen()

            asyncio.create_task(do_export())
        elif event.button.id == "cancel":
            self.app.pop_screen()
