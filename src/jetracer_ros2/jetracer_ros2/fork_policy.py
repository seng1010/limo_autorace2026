"""Branch policy; perception must explicitly report entry and exit."""


class ForkPolicy:
    def __init__(self, default='stop'):
        if default not in ('stop', 'left', 'right'):
            raise ValueError('fork_default must be stop, left or right')
        self.default = default
        self.active = False
        self.requested = None

    @property
    def direction(self):
        return self.requested or (None if self.default == 'stop' else self.default)

    def event(self, event):
        if event == 'FORK_ENTER':
            self.active = True
        elif event in ('FORK_LEFT', 'FORK_RIGHT'):
            self.requested = event[5:].lower()
        elif event == 'FORK_EXIT':
            self.active = False
            self.requested = None
        else:
            return False
        return True
