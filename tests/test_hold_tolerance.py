from visual_actions.core.bindings import Bindings
from visual_actions.core.events import Bus
from visual_actions.core.modes import ARMED, HOLDING, IDLE, ModeEngine, Timing
from visual_actions.core.types import Hand, Token

S = 1_000_000_000


def make(break_tokens=2):
    eng = ModeEngine(Bus(), Bindings(), Timing(leader_hold_ns=2 * S, confidence_gain=1.0, hold_break_tokens=break_tokens), fire=lambda a, t: None)
    return eng


def palm(eng, t, conf=1.0):
    eng.on_token(Token(int(t * S), "open_palm", conf, Hand.RIGHT, True))


def test_single_flicker_pauses_the_hold_instead_of_breaking_it():
    eng = make()
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        palm(eng, t)
    eng.on_token(Token(int(1.25 * S), "none", 0.5, Hand.RIGHT, True))  # one flicker
    assert eng.state == HOLDING
    for t in (1.5, 1.75, 2.0, 2.25, 2.5, 2.75):
        palm(eng, t)
    eng.on_tick(int(2.8 * S))
    assert eng.state == ARMED  # 1.0 s before the flicker + ~1.25 s after (the gap itself adds nothing)


def test_two_flickers_in_a_row_break_the_hold():
    eng = make()
    palm(eng, 0.0)
    palm(eng, 0.25)
    eng.on_token(Token(int(0.5 * S), "none", 0.5, Hand.RIGHT, True))
    eng.on_token(Token(int(0.75 * S), "fist", 0.5, Hand.RIGHT, True))
    assert eng.state == IDLE


def test_flicker_counter_resets_on_a_palm():
    eng = make()
    palm(eng, 0.0)
    eng.on_token(Token(int(0.25 * S), "none", 0.5, Hand.RIGHT, True))
    palm(eng, 0.5)
    eng.on_token(Token(int(0.75 * S), "none", 0.5, Hand.RIGHT, True))
    assert eng.state == HOLDING
