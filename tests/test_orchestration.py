from __future__ import annotations

from src.system.orchestration import TaskDefinition, WorkflowOrchestrator


def test_orchestrator_respects_dependencies() -> None:
    order: list[str] = []
    orchestrator = WorkflowOrchestrator(
        {
            "ingest": TaskDefinition(lambda: order.append("ingest") or {"rows": 10}),
            "train": TaskDefinition(
                lambda: order.append("train") or {"model": "m"}, dependencies=("ingest",)
            ),
        }
    )
    result = orchestrator.execute("pipeline")
    assert result["status"] == "success"
    assert order == ["ingest", "train"]


def test_orchestrator_skips_dependents_after_failure() -> None:
    def fail() -> dict[str, object]:
        raise RuntimeError("boom")

    orchestrator = WorkflowOrchestrator(
        {
            "ingest": TaskDefinition(fail),
            "train": TaskDefinition(lambda: {}, dependencies=("ingest",)),
        }
    )
    result = orchestrator.execute("pipeline")
    assert result["status"] == "failed"
    assert result["tasks"]["train"]["status"] == "skipped"
