from src.tools.note_taker import NoteStore, create_note_tools


def test_note_store_add_and_get():
    store = NoteStore()
    result = store.add("First finding")
    assert "1" in result
    assert store.as_list() == ["First finding"]


def test_note_store_get_all_empty():
    store = NoteStore()
    assert "No notes" in store.get_all()


def test_note_store_get_all_with_notes():
    store = NoteStore()
    store.add("Note A")
    store.add("Note B")
    output = store.get_all()
    assert "1. Note A" in output
    assert "2. Note B" in output


def test_note_store_clear():
    store = NoteStore()
    store.add("temp")
    store.clear()
    assert store.as_list() == []


def test_create_note_tools():
    store = NoteStore()
    tools = create_note_tools(store)
    assert len(tools) == 2
    assert tools[0].name == "add_note"
    assert tools[1].name == "get_notes"
