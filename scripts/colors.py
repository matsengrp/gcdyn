import collections

# ----------------------------------------------------------------------------------------
# bash color codes
ansi_color_table = collections.OrderedDict((
    # 'head' : '95'  # not sure wtf this was?
    ('end', 0),
    ('bold', 1),
    ('reverse_video', 7),
    ('grey', 90),
    ('red', 91),
    ('green', 92),
    ('yellow', 93),
    ('blue', 94),
    ('purple', 95),
    ('grey_bkg', 100),
    ('red_bkg', 41),
    ('green_bkg', 42),
    ('yellow_bkg', 43),
    ('blue_bkg', 44),
    ('light_blue_bkg', 104),
    ('purple_bkg', 45),
))
Colors = {c : '\033[%sm' % i for c, i in ansi_color_table.items()}


# ----------------------------------------------------------------------------------------
def color(col, seq, width=None, padside='left'):
    return_str = [seq]
    if col is not None:
        return_str = [Colors[col]] + return_str + [Colors['end']]
    if width is not None:  # make sure final string prints to correct width
        n_spaces = max(0, width - len(seq))  # if specified <width> is greater than uncolored length of <seq>, pad with spaces so that when the colors show up properly the colored sequences prints with width <width>
        if padside == 'left':
            return_str.insert(0, n_spaces * ' ')
        elif padside == 'right':
            return_str.insert(len(return_str), n_spaces * ' ')
        else:
            assert False
    return ''.join(return_str)
