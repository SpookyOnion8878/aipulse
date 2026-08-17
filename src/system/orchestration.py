"""Small in-process workflow orchestrator with explicit dependencies."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

TaskAction = Callable[[], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    action: TaskAction
    dependencies: tuple[str, ...] = ()


@dataclass(slots=True)
class TaskResult:
    name: str
    status: str
    duration_seconds: float
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class WorkflowOrchestrator:
    """Execute a dependency graph once, failing dependents safely."""

    def __init__(self, tasks: dict[str, TaskDefinition]) -> None:
        self.tasks = tasks
        self._validate_graph()

    def _validate_graph(self) -> None:
        for name, task in self.tasks.items():
            missing = sorted(set(task.dependencies) - set(self.tasks))
            if missing:
                raise ValueError(f"Task {name} has unknown dependencies: {missing}")

        visited: set[str] = set()
        active: set[str] = set()

        def visit(name: str) -> None:
            if name in active:
                raise ValueError(f"Dependency cycle detected at {name}")
            if name in visited:
                return
            active.add(name)
            for dependency in self.tasks[name].dependencies:
                visit(dependency)
            active.remove(name)
            visited.add(name)

        for name in self.tasks:
            visit(name)

    def execute(self, workflow_name: str) -> dict[str, Any]:
        started = time.monotonic()
        results: dict[str, TaskResult] = {}
        pending = set(self.tasks)
        while pending:
            progress = False
            for name in sorted(pending):
                task = self.tasks[name]
                if not all(dependency in results for dependency in task.dependencies):
                    continue
                failed_dependency = next(
                    (
                        dependency
                        for dependency in task.dependencies
                        if results[dependency].status != "success"
                    ),
                    None,
                )
                if failed_dependency is not None:
                    results[name] = TaskResult(
                        name=name,
                        status="skipped",
                        duration_seconds=0.0,
                        error=f"Dependency failed: {failed_dependency}",
                    )
                else:
                    task_started = time.monotonic()
                    try:
                        output = task.action()
                    except Exception as exc:
                        results[name] = TaskResult(
                            name=name,
                            status="failed",
                            duration_seconds=time.monotonic() - task_started,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    else:
                        results[name] = TaskResult(
                            name=name,
                            status="success",
                            duration_seconds=time.monotonic() - task_started,
                            output=output,
                        )
                pending.remove(name)
                progress = True
            if not progress:
                raise RuntimeError("Workflow cannot make progress")
        workflow_status = (
            "success" if all(item.status == "success" for item in results.values()) else "failed"
        )
        return {
            "workflow_name": workflow_name,
            "status": workflow_status,
            "duration_seconds": time.monotonic() - started,
            "finished_at": datetime.now(UTC).isoformat(),
            "tasks": {name: asdict(item) for name, item in results.items()},
        }
