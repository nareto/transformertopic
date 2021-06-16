import hashlib

def generateTextId(text: str):
    return hashlib.md5(text.encode('utf-8')).hexdigest()