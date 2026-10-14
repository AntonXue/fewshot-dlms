"""Test doubles that let the CPU tests run without a model or a tokenizer download."""


class FakeAdapter:
    """A stand-in for a model adapter: byte-level tokenizer, fixed mask id, short chat markers.

    It has the token-level interface that `canvas.build_canvas` needs from a real adapter:
    `encode`, `mask_id` and the four chat-framing methods. Every character is its own token (its
    character code), so a test can read a canvas back as text with `decode`. The mask id 256 lies
    outside the byte range, so it can never collide with a real token.
    """

    name = "fake"
    mask_id = 256

    def encode(self, text):
        """One token per character: the character code."""
        return [ord(character) for character in text]

    def decode(self, ids, mask="_"):
        """Read token ids back as text, writing `mask` for every mask token."""
        return "".join(mask if token == self.mask_id else chr(token) for token in ids)

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
