from __future__ import annotations

from collections.abc import Iterable

from .types import Action, Binding


class Bindings:
    def __init__(self, bindings: Iterable[Binding] = ()) -> None:
        self._table: dict[tuple[str, str], Action] = {}
        for b in bindings:
            self.add(b)

    def add(self, b: Binding) -> None:
        self._table[(b.namespace, b.gesture)] = b.action

    def lookup(self, namespace: str, gesture: str) -> Action | None:
        return self._table.get((namespace, gesture))

    def namespaces(self) -> set[str]:
        return {ns for ns, _ in self._table}

    def __len__(self) -> int:
        return len(self._table)
