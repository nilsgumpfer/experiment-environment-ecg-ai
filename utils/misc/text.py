def convert_underscored_to_camelcaps(text):
    splits = str(text).split('_')
    finalword = ''

    for part in splits:
        finalword += str(part).capitalize()

    return finalword
