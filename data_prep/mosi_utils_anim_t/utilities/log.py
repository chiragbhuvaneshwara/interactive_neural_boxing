LOG_MODE_ERROR = -1
LOG_MODE_INFO = 1
LOG_MODE_DEBUG = 2

_lines = []
_active = True
_mode = LOG_MODE_INFO

def activate():
    global _active
    _active = True


def deactivate():
    global _active
    _active = False

def set_log_mode(mode):
    global _mode
    _mode = mode

def write_log(*args):
    global _active
    global _lines
    if _active:
        line = " ".join(map(str, args))
        print(line)
        _lines.append(line)


def write_message_to_log(message, mode=LOG_MODE_INFO):
    global _active
    global _lines
    if _active and _mode >= mode:
        print(message)
        _lines.append(message)


def save_log(filename):
    global _lines
    with open(filename, "wb") as outfile:
        for l in _lines:
            outfile.write(l+"\n")


def clear_log():
    global _lines
    _lines = []
