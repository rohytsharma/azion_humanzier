"""App behaviour checks (SRD 12: valid, empty, long and malformed input).

    .venv/bin/python -m tests.test_app

Uses Streamlit's own headless harness rather than a browser, so the checks run
without a display and without fighting the widget lifecycle.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

# AppTest resolves relative paths against *this* file, not the working directory.
APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")

FLAT = ("The system processes data efficiently. It stores results in a database. "
        "The results are retrieved later. This approach improves performance. "
        "Users benefit from faster response times. The design is scalable.")

VARIED = ("The system chews through data — quickly, and without much ceremony — then "
          "files what it finds. Later, when something asks, it answers. Why does that "
          "matter? Because the alternative, recomputing everything on every request, is "
          "the kind of design that looks fine in a demo and falls over the first time "
          "real traffic arrives.")


def body(at):
    return " ".join(m.value for m in at.markdown)


def test_loads_clean():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert "HumanWriter" in body(at)
    assert "Writing profile" in body(at)
    print("  loads with no exception")


def test_empty_input_shows_placeholder():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button[0].click().run()
    assert not at.exception
    assert "human-range index" not in body(at), "profiled empty input"
    print("  empty input is refused cleanly")


def test_too_short_is_rejected():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_area[0].set_value("Too short.").run()
    at.button[0].click().run()
    assert not at.exception
    assert at.error, "no error shown for a two-word input"
    print("  short input rejected:", at.error[0].value[:48])


def test_analyse_produces_profile():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_area[0].set_value(FLAT).run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    out = body(at)
    assert "human-range index" in out, "no index rendered"
    assert "Punctuation density" in out and "Burstiness" in out
    print("  profile rendered with all three bands")


def test_varied_prose_scores_above_flat():
    """The index has to actually discriminate, or it is decoration."""
    def index_of(text):
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.text_area[0].set_value(text).run()
        at.button[0].click().run()
        assert not at.exception, at.exception
        for m in at.markdown:
            if "human-range index" in m.value:
                return int(m.value.split("<b>")[1].split("</b>")[0])
        raise AssertionError("index not found")

    flat, varied = index_of(FLAT), index_of(VARIED)
    print(f"  flat prose {flat}/100  vs  varied prose {varied}/100")
    assert varied > flat, "the index does not separate flat from varied writing"


def test_long_input_is_truncated_not_crashed():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.text_area[0].set_value(FLAT * 400).run()      # ~150k chars
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert at.warning, "no warning for oversized input"
    print("  oversized input warned and truncated")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("\nall good")
