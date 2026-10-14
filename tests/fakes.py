"""Test doubles that let the CPU tests run without a model or a tokenizer download."""


class FakeAdapter:
    """A stand-in for a model adapter: byte-level tokenizer, fixed mask id, short chat markers.

    It has the token-level interface that `canvas.build_canvas` needs from a real adapter:
    `encode`, `mask_id` and the four chat-framing methods. A token is one byte of the UTF-8 text,
    so plain ASCII text has one token per character and a test can read a canvas back as text with
    `decode`. Byte values are 0..255, so the mask id 256 can never collide with a text token.
    """

    name = "fake"
    mask_id = 256

    def encode(self, text):
        """The UTF-8 bytes of the text, one token per byte."""
        return list(text.encode("utf-8"))

    def decode(self, ids, mask="_"):
        """Read token ids back as text, writing `mask` for every mask token."""
        pieces = []
        pending = bytearray()
        for token in ids:
            if token == self.mask_id:
                pieces.append(bytes(pending).decode("utf-8", errors="replace"))
                pieces.append(mask)
                pending = bytearray()
            else:
                pending.append(token)
        pieces.append(bytes(pending).decode("utf-8", errors="replace"))
        return "".join(pieces)

    def chat_header(self, instruction):
        """Everything before the first record: a header marker and the instruction."""
        return "<h>" + instruction

    def chat_user_open(self):
        """Opens the user turn (used by the single-target canvas, which is not built yet)."""
        return "<u>"

    def chat_user_close_assistant_open(self):
        """Closes the user turn and opens the assistant turn (single-target canvas)."""
        return "<a>"

    def chat_end(self):
        """Ends the assistant turn; the last token of every canvas."""
        return "<e>"
