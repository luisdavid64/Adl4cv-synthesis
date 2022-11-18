
import re

def lower_slash_format(s):
    return s.lower().replace(" / ", "/") 

def normalize_rgb(color):
    return color/255
    