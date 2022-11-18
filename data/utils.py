
import re

def lower_slash_format(s):
    return re.sub(r"\s[/]\s", "/", s.lower())

def normalize_rgb(color):
    return color/255
    