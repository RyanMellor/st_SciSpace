
import streamlit as st
import numpy as np
from skimage import color

colors_name_hex = {
    'AliceBlue': 	'#F0F8FF',
    'AntiqueWhite': 	'#FAEBD7',
    'Aqua': 	'#00FFFF',
    'Aquamarine': 	'#7FFFD4',
    'Azure': 	'#F0FFFF',
    'Beige': 	'#F5F5DC',
    'Bisque': 	'#FFE4C4',
    'Black': 	'#000000',
    'BlanchedAlmond': 	'#FFEBCD',
    'Blue': 	'#0000FF',
    'BlueViolet': 	'#8A2BE2',
    'Brown': 	'#A52A2A',
    'BurlyWood': 	'#DEB887',
    'CadetBlue': 	'#5F9EA0',
    'Chartreuse': 	'#7FFF00',
    'Chocolate': 	'#D2691E',
    'Coral': 	'#FF7F50',
    'CornflowerBlue': 	'#6495ED',
    'Cornsilk': 	'#FFF8DC',
    'Crimson': 	'#DC143C',
    'Cyan': 	'#00FFFF',
    'DarkBlue': 	'#00008B',
    'DarkCyan': 	'#008B8B',
    'DarkGoldenRod': 	'#B8860B',
    'DarkGray': 	'#A9A9A9',
    'DarkGrey': 	'#A9A9A9',
    'DarkGreen': 	'#006400',
    'DarkKhaki': 	'#BDB76B',
    'DarkMagenta': 	'#8B008B',
    'DarkOliveGreen': 	'#556B2F',
    'Darkorange': 	'#FF8C00',
    'DarkOrchid': 	'#9932CC',
    'DarkRed': 	'#8B0000',
    'DarkSalmon': 	'#E9967A',
    'DarkSeaGreen': 	'#8FBC8F',
    'DarkSlateBlue': 	'#483D8B',
    'DarkSlateGray': 	'#2F4F4F',
    'DarkSlateGrey': 	'#2F4F4F',
    'DarkTurquoise': 	'#00CED1',
    'DarkViolet': 	'#9400D3',
    'DeepPink': 	'#FF1493',
    'DeepSkyBlue': 	'#00BFFF',
    'DimGray': 	'#696969',
    'DimGrey': 	'#696969',
    'DodgerBlue': 	'#1E90FF',
    'FireBrick': 	'#B22222',
    'FloralWhite': 	'#FFFAF0',
    'ForestGreen': 	'#228B22',
    'Fuchsia': 	'#FF00FF',
    'Gainsboro': 	'#DCDCDC',
    'GhostWhite': 	'#F8F8FF',
    'Gold': 	'#FFD700',
    'GoldenRod': 	'#DAA520',
    'Gray': 	'#808080',
    'Grey': 	'#808080',
    'Green': 	'#008000',
    'GreenYellow': 	'#ADFF2F',
    'HoneyDew': 	'#F0FFF0',
    'HotPink': 	'#FF69B4',
    'IndianRed ': 	'#CD5C5C',
    'Indigo ': 	'#4B0082',
    'Ivory': 	'#FFFFF0',
    'Khaki': 	'#F0E68C',
    'Lavender': 	'#E6E6FA',
    'LavenderBlush': 	'#FFF0F5',
    'LawnGreen': 	'#7CFC00',
    'LemonChiffon': 	'#FFFACD',
    'LightBlue': 	'#ADD8E6',
    'LightCoral': 	'#F08080',
    'LightCyan': 	'#E0FFFF',
    'LightGoldenRodYellow': 	'#FAFAD2',
    'LightGray': 	'#D3D3D3',
    'LightGrey': 	'#D3D3D3',
    'LightGreen': 	'#90EE90',
    'LightPink': 	'#FFB6C1',
    'LightSalmon': 	'#FFA07A',
    'LightSeaGreen': 	'#20B2AA',
    'LightSkyBlue': 	'#87CEFA',
    'LightSlateGray': 	'#778899',
    'LightSlateGrey': 	'#778899',
    'LightSteelBlue': 	'#B0C4DE',
    'LightYellow': 	'#FFFFE0',
    'Lime': 	'#00FF00',
    'LimeGreen': 	'#32CD32',
    'Linen': 	'#FAF0E6',
    'Magenta': 	'#FF00FF',
    'Maroon': 	'#800000',
    'MediumAquaMarine': 	'#66CDAA',
    'MediumBlue': 	'#0000CD',
    'MediumOrchid': 	'#BA55D3',
    'MediumPurple': 	'#9370D8',
    'MediumSeaGreen': 	'#3CB371',
    'MediumSlateBlue': 	'#7B68EE',
    'MediumSpringGreen': 	'#00FA9A',
    'MediumTurquoise': 	'#48D1CC',
    'MediumVioletRed': 	'#C71585',
    'MidnightBlue': 	'#191970',
    'MintCream': 	'#F5FFFA',
    'MistyRose': 	'#FFE4E1',
    'Moccasin': 	'#FFE4B5',
    'NavajoWhite': 	'#FFDEAD',
    'Navy': 	'#000080',
    'OldLace': 	'#FDF5E6',
    'Olive': 	'#808000',
    'OliveDrab': 	'#6B8E23',
    'Orange': 	'#FFA500',
    'OrangeRed': 	'#FF4500',
    'Orchid': 	'#DA70D6',
    'PaleGoldenRod': 	'#EEE8AA',
    'PaleGreen': 	'#98FB98',
    'PaleTurquoise': 	'#AFEEEE',
    'PaleVioletRed': 	'#D87093',
    'PapayaWhip': 	'#FFEFD5',
    'PeachPuff': 	'#FFDAB9',
    'Peru': 	'#CD853F',
    'Pink': 	'#FFC0CB',
    'Plum': 	'#DDA0DD',
    'PowderBlue': 	'#B0E0E6',
    'Purple': 	'#800080',
    'Red': 	'#FF0000',
    'RosyBrown': 	'#BC8F8F',
    'RoyalBlue': 	'#4169E1',
    'SaddleBrown': 	'#8B4513',
    'Salmon': 	'#FA8072',
    'SandyBrown': 	'#F4A460',
    'SeaGreen': 	'#2E8B57',
    'SeaShell': 	'#FFF5EE',
    'Sienna': 	'#A0522D',
    'Silver': 	'#C0C0C0',
    'SkyBlue': 	'#87CEEB',
    'SlateBlue': 	'#6A5ACD',
    'SlateGray': 	'#708090',
    'SlateGrey': 	'#708090',
    'Snow': 	'#FFFAFA',
    'SpringGreen': 	'#00FF7F',
    'SteelBlue': 	'#4682B4',
    'Tan': 	'#D2B48C',
    'Teal': 	'#008080',
    'Thistle': 	'#D8BFD8',
    'Tomato': 	'#FF6347',
    'Turquoise': 	'#40E0D0',
    'Violet': 	'#EE82EE',
    'Wheat': 	'#F5DEB3',
    'White': 	'#FFFFFF',
    'WhiteSmoke': 	'#F5F5F5',
    'Yellow': 	'#FFFF00',
    'YellowGreen': 	'#9ACD32',
}


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))


# def rgb_to_lab(inputColor):
#     num = 0
#     RGB = [0, 0, 0]

#     for value in inputColor:
#         value = float(value) / 255

#         if value > 0.04045:
#             value = ((value + 0.055) / 1.055) ** 2.4
#         else:
#             value = value / 12.92

#         RGB[num] = value * 100
#         num = num + 1

#     XYZ = [0, 0, 0,]

#     X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
#     Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
#     Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
#     XYZ[0] = round(X, 4)
#     XYZ[1] = round(Y, 4)
#     XYZ[2] = round(Z, 4)

#     # ref_X =  95.047   Observer= 2°, Illuminant= D65
#     XYZ[0] = float(XYZ[0]) / 95.047
#     XYZ[1] = float(XYZ[1]) / 100.0          # ref_Y = 100.000
#     XYZ[2] = float(XYZ[2]) / 108.883        # ref_Z = 108.883

#     num = 0
#     for value in XYZ:
#         if value > 0.008856:
#             value = value ** (0.3333333333333333)
#         else:
#             value = (7.787 * value) + (16 / 116)

#         XYZ[num] = value
#         num = num + 1

#     Lab = [0, 0, 0]

#     L = (116 * XYZ[1]) - 16
#     a = 500 * (XYZ[0] - XYZ[1])
#     b = 200 * (XYZ[1] - XYZ[2])

#     Lab[0] = round(L, 4)
#     Lab[1] = round(a, 4)
#     Lab[2] = round(b, 4)

#     return Lab


def color_difference(color1, color2):
    return np.sqrt(np.sum((np.array(color1)-np.array(color2))**2))


def nearest_named_color(hex):
    color_lab = color.rgb2lab(hex_to_rgb(hex))
    nearest_color_name = min(colors_name_hex.keys(), key=lambda name: color_difference(
        color_lab, color.rgb2lab(hex_to_rgb(colors_name_hex[name]))))
    nearest_color_hex = min(colors_name_hex.values(
    ), key=lambda rgb: color_difference(color_lab, color.rgb2lab(hex_to_rgb(rgb))))
    return nearest_color_name, nearest_color_hex


def make_grid(rows, cols):
    grid = [0]*rows
    for i in range(rows):
        with st.container():
            grid[i] = st.columns(cols)
    return grid
