"""Branch policy; perception must explicitly report entry and exit."""


class ForkPolicy:
    def __init__(self, default='stop'):
        if default not in ('stop', 'left', 'right'):
            raise ValueError('fork_default must be stop, left or right')
        self.default = default
        self.active = False
        self.requested = None
        self.position = None  # Physical lane position, not a boundary-tracking mode.
        self.stage = None

    @property
    def direction(self):
        return self.requested or (None if self.default == 'stop' else self.default)

    @property
    def mode(self):
        """Requested boundary mode; None means the manager must stop."""
        if not self.active or self.direction is None or self.position is None:
            return None
        return self.stage or self.direction

    def _choose_stage(self):
        direction = self.direction
        if self.active and direction and self.position:
            if self.position in ('left', 'right') and self.position != direction:
                self.stage = 'center'
            else:
                self.stage = direction
        else:
            self.stage = None

    def event(self, event):
        if event == 'FORK_ENTER':
            self.active = True
            self._choose_stage()
        elif event in ('FORK_LEFT', 'FORK_RIGHT'):
            self.requested = event[5:].lower()
            self._choose_stage()
        elif event in ('FORK_POSITION_LEFT', 'FORK_POSITION_CENTER',
                       'FORK_POSITION_RIGHT'):
            self.position = event[len('FORK_POSITION_'):].lower()
            self._choose_stage()
        elif event == 'FORK_CENTER_REACHED':
            if not self.active or self.stage != 'center':
                return False
            self.position = 'center'
            self._choose_stage()
        elif event == 'FORK_EXIT':
            self.active = False
            self.requested = None
            self.position = None
            self.stage = None
        else:
            return False
        return True
