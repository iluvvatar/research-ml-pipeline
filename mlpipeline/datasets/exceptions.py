class FormatError(Exception):
    def __init__(self, msg: str = None, full_msg: bool = False):
        if full_msg:
            super().__init__(f'Incorrect format: {msg}')
        else:
            super().__init__(msg)


class BratFormatError(FormatError):
    def __init__(self, msg):
        super().__init__(f'Incorrect BRAT format: {msg}', True)
