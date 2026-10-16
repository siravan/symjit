class Assembler:
    def __init__(self):
        self.buf = bytearray()
        self.tail = None
        self.labels = {}
        self.jumps = []

    def append_byte(self, *b):
        self.buf.extend(b)

    def append_word(self, u):
        """appends u (uint32) as little-endian"""
        self.buf.append(u & 0xFF)
        self.buf.append((u >> 8) & 0xFF)
        self.buf.append((u >> 16) & 0xFF)
        self.buf.append((u >> 24) & 0xFF)

    def begin_prepend(self):
        assert self.tail is None
        self.tail = self.buf
        self.buf = bytearray()

    def end_prepend(self):
        assert self.tail is not None
        shift = len(self.buf)
        self.buf.extend(self.tail)
        self.tail = None
        # note: prologue cannot have jumps
        jumps = []
        for label, k in self.jumps:
            jumps.append((label, k + shift))
        self.jumps = jumps

    def set_label(self, label):
        self.labels[label] = len(self.buf)

    def jump(self, label, code=0):
        self.jumps.append((label, len(self.buf)))
        self.append_word(code)

    def test(self, b):
        if self.buf == bytes(b):
            print(f"pass: {b}")
            self.buf.clear()
            return True
        else:
            return False
