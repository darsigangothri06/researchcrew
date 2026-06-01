from langchain_core.tools import tool


class NoteStore:
    """Per-session note storage for research findings."""

    def __init__(self):
        self._notes: list[str] = []

    def add(self, note: str) -> str:
        self._notes.append(note)
        return f"Note saved. Total notes: {len(self._notes)}"

    def get_all(self) -> str:
        if not self._notes:
            return "No notes saved yet."
        return "\n".join(f"{i + 1}. {n}" for i, n in enumerate(self._notes))

    def as_list(self) -> list[str]:
        return list(self._notes)

    def clear(self):
        self._notes.clear()


def create_note_tools(store: NoteStore):
    @tool
    def add_note(note: str) -> str:
        """Save an important finding or observation for later synthesis."""
        return store.add(note)

    @tool
    def get_notes() -> str:
        """Retrieve all saved research notes."""
        return store.get_all()

    return [add_note, get_notes]
