"""Plain, transparent rules applied after handwriting recognition."""

FORMATS = {
    'General number': ('Any 1–12 digit number.', lambda text: 1 <= len(text) <= 12, None),
    'Indian mobile number': ('Exactly 10 digits; first digit must be 6, 7, 8 or 9.',
                              lambda text: len(text) == 10 and text[0] in '6789', 'XXXXXXXXXX'),
    'Indian PIN code': ('Exactly 6 digits; first digit must be 1–8.',
                        lambda text: len(text) == 6 and text[0] in '12345678', 'XXXXXX'),
    'OTP (6 digits)': ('Exactly 6 digits.', lambda text: len(text) == 6, 'XXXXXX'),
    'Student/roll number': ('Exactly 8 digits.', lambda text: len(text) == 8, 'XXXXXXXX'),
}


def validate_format(text, format_name):
    """Return a readable rule result. Recognition and format validity remain separate."""
    if format_name not in FORMATS:
        raise ValueError('Unknown format.')
    explanation, rule, mask = FORMATS[format_name]
    if text is None:
        return {'valid': False, 'message': 'No number was safely recognised yet.', 'mask': mask}
    valid = bool(rule(text))
    return {'valid': valid,
            'message': 'Matches the selected format.' if valid else f'Format rule not met: {explanation}',
            'mask': mask}
