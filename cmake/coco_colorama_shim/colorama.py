def init(*args, **kwargs):
    return None


class _Ansi:
    def __getattr__(self, name):
        return ""


Fore = _Ansi()
Style = _Ansi()
