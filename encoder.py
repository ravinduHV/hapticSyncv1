phoneme_to_ipa_braille = {
    "aa": {"ipa": "ɑ", "braille": "⠷", "binary": "111011"},
    "ae": {"ipa": "æ", "braille": "⠮", "binary": "111010"},
    "ah": {"ipa": "ʌ", "braille": "⠬", "binary": "111100"},
    "aw": {"ipa": "aʊ", "braille": "⠷⠬", "binary": ["111011", "111100"]},
    "ay": {"ipa": "aɪ", "braille": "⠷⠊", "binary": ["111011", "010010"]},
    "b":  {"ipa": "b", "braille": "⠃", "binary": "100001"},
    "ch": {"ipa": "tʃ", "braille": "⠩", "binary": "101001"},
    "d":  {"ipa": "d", "braille": "⠙", "binary": "101001"},
    "dh": {"ipa": "ð", "braille": "⠹", "binary": "111001"},
    "dx": {"ipa": "ɾ", "braille": "⠄", "binary": "000100"},
    "eh": {"ipa": "ɛ", "braille": "⠣", "binary": "110001"},
    "er": {"ipa": "ɝ", "braille": "⠱", "binary": "111000"},
    "ey": {"ipa": "eɪ", "braille": "⠣⠊", "binary": ["110001", "010010"]},
    "f":  {"ipa": "f", "braille": "⠋", "binary": "110001"},
    "g":  {"ipa": "g", "braille": "⠛", "binary": "110011"},
    "h#": {"ipa": "sil", "braille": "⠄", "binary": "000100"},
    "hh": {"ipa": "h", "braille": "⠓", "binary": "110010"},
    "ih": {"ipa": "ɪ", "braille": "⠊", "binary": "010010"},
    "iy": {"ipa": "i", "braille": "⠊", "binary": "010010"},
    "jh": {"ipa": "dʒ", "braille": "⠫", "binary": "101011"},
    "k":  {"ipa": "k", "braille": "⠅", "binary": "100101"},
    "l":  {"ipa": "l", "braille": "⠇", "binary": "100111"},
    "m":  {"ipa": "m", "braille": "⠍", "binary": "101101"},
    "n":  {"ipa": "n", "braille": "⠝", "binary": "101111"},
    "ng": {"ipa": "ŋ", "braille": "⠻", "binary": "111011"},
    "ow": {"ipa": "oʊ", "braille": "⠕⠬", "binary": ["101101", "111100"]},
    "oy": {"ipa": "ɔɪ", "braille": "⠪⠊", "binary": ["101010", "010010"]},
    "p":  {"ipa": "p", "braille": "⠏", "binary": "110111"},
    "r":  {"ipa": "ɹ", "braille": "⠗", "binary": "111001"},
    "s":  {"ipa": "s", "braille": "⠎", "binary": "110110"},
    "sh": {"ipa": "ʃ", "braille": "⠩", "binary": "101001"},
    "t":  {"ipa": "t", "braille": "⠞", "binary": "111010"},
    "th": {"ipa": "θ", "braille": "⠹", "binary": "111001"},
    "uh": {"ipa": "ʊ", "braille": "⠳", "binary": "111011"},
    "uw": {"ipa": "u", "braille": "⠥", "binary": "101011"},
    "v":  {"ipa": "v", "braille": "⠧", "binary": "101111"},
    "w":  {"ipa": "w", "braille": "⠺", "binary": "110111"},
    "y":  {"ipa": "j", "braille": "⠽", "binary": "111101"},
    "z":  {"ipa": "z", "braille": "⠵", "binary": "111110"},
    "|":  {"ipa": "|", "braille": "⠸", "binary": "111000"},
    "[UNK]": {"ipa": "?", "braille": "⠿", "binary": "111111"},
    "[PAD]": {"ipa": "", "braille": "⠀", "binary": "000000"},
}


def get_ipa_braille(phoneme):
    """
    Given a phoneme, return its IPA and Braille representation.
    """
    if phoneme in phoneme_to_ipa_braille:
        return phoneme_to_ipa_braille[phoneme]["braille"]
    else:
        return None, None


def get_ipa(phoneme):
    """
    Given a phoneme, return its IPA representation.
    """
    if phoneme in phoneme_to_ipa_braille:
        return phoneme_to_ipa_braille[phoneme]["ipa"]
    else:
        return None
     

def get_binary(phoneme):
    """
    Given a phoneme, return its binary representation.
    """
    if phoneme in phoneme_to_ipa_braille:
        binary = phoneme_to_ipa_braille[phoneme]["binary"]
        if isinstance(binary, list):
            return " ".join(binary)
        return binary
    else:
        return None
