import platform

def getInstalledFontPath():
    os = platform.system()
    if os == 'Linux':
        return '/usr/share/fonts/truetype/freefont/FreeSans.ttf'
    elif os == 'Darwin':
        return '/Library/Fonts/Arial.ttf'
    elif os == 'Windows':
        return 'C:\\Windows\\Fonts\\Arial.ttf'
    else:
        return 'Arial.ttf'

