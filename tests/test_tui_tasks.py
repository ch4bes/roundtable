"""Tests for TUI fire-and-forget task error handling (fix 1.6)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tui.app import RoundtableApp


class MockApp:
    """Minimal mock of RoundtableApp for testing _safe_create_task."""

    def __init__(self):
        self._tasks: set[asyncio.Task] = set()
        self.is_running = True
        self._notifications = []

    def notify(self, message: str, severity: str | None = None):
        self._notifications.append((message, severity))


@pytest.fixture
def mock_app():
    app = MockApp()
    # Bind _safe_create_task from RoundtableApp onto the mock
    app._safe_create_task = RoundtableApp._safe_create_task.__get__(app, MockApp)
    return app


async def _succeed():
    return "ok"


async def _fail():
    raise RuntimeError("simulated error")


async def _slow():
    await asyncio.sleep(10)


@pytest.mark.asyncio
class TestSafeCreateTaskSuccess:
    async def test_safe_create_task_handles_success(self, mock_app):
        """_safe_create_task notifies on success."""
        task = mock_app._safe_create_task(_succeed(), "Done", "Failed")
        assert task in mock_app._tasks

        await asyncio.wait_for(task, timeout=2)

        assert len(mock_app._notifications) == 1
        assert mock_app._notifications[0] == ("Done", None)


@pytest.mark.asyncio
class TestSafeCreateTaskError:
    async def test_safe_create_task_handles_error(self, mock_app):
        """_safe_create_task notifies on error, doesn't raise."""
        task = mock_app._safe_create_task(_fail(), "Done", "Failed")

        await asyncio.wait_for(task, timeout=2)

        assert len(mock_app._notifications) == 1
        msg, severity = mock_app._notifications[0]
        assert "Failed: simulated error" == msg
        assert severity == "error"


@pytest.mark.asyncio
class TestSafeCreateTaskTracking:
    async def test_safe_create_task_adds_to_tasks_set(self, mock_app):
        """Task is added to _tasks set immediately."""
        task = mock_app._safe_create_task(_succeed(), "Done", "Failed")
        assert task in mock_app._tasks

    async def test_safe_create_task_removes_on_completion(self, mock_app):
        """Task is removed from _tasks set when done."""
        task = mock_app._safe_create_task(_succeed(), "Done", "Failed")
        assert task in mock_app._tasks

        await asyncio.wait_for(task, timeout=2)

        assert task not in mock_app._tasks


@pytest.mark.asyncio
class TestOnExitCancelsPendingTasks:
    async def test_on_exit_cancels_pending_tasks(self):
        """on_exit cancels all pending tasks in _tasks."""
        app = RoundtableApp()
        app._tasks = set()

        # Create a long-running task and add it
        task = asyncio.create_task(_slow())
        app._tasks.add(task)

        assert not task.done()
        app.on_exit()

        await asyncio.sleep(0.01)   # let cancel propagate
        assert task.cancelled() or task.done()
        assert len(app._tasks) == 0


@pytest.mark.asyncio
class TestSafeCreateTaskNoNotifyWhenNotRunning:
    async def test_safe_create_task_no_notify_if_not_running(self, mock_app):
        """No notification if app is not running when task completes."""
        mock_app.is_running = False
        task = mock_app._safe_create_task(_succeed(), "Done", "Failed")

        await asyncio.wait_for(task, timeout=2)

        assert len(mock_app._notifications) == 0

    async def test_safe_create_task_no_error_notify_if_not_running(self, mock_app):
        """No error notification if app is not running when task fails."""
        mock_app.is_running = False
        task = mock_app._safe_create_task(_fail(), "Done", "Failed")

        await asyncio.wait_for(task, timeout=2)

        assert len(mock_app._notifications) == 0


@pytest.mark.asyncio
class TestSafeCreateTaskCancelledError:
    async def test_safe_create_task_handles_cancelled_error(self, mock_app):
        """CancelledError doesn't produce an error notification."""
        task = mock_app._safe_create_task(_slow(), "Done", "Failed")

        # Let it start, then cancel it
        await asyncio.sleep(0.01)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have no notifications (CancelledError is handled silently)
        assert len(mock_app._notifications) == 0
