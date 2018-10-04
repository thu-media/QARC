import os


def main():
    _dict = {}
    _files = os.listdir('img/')
    for _file in _files:
        _split = _file.split('_')
        _filename = _split[0]
        _png = _split[1]
        if _dict.has_key(_filename):
            _dict[_filename].append(_png)
        else:
            _dict[_filename] = []
    for _filename, _detail in _dict.items():
        os.system('mkdir img2/' + _filename)
        for p in _detail:
            os.system('cp -f img/' + _filename + '_' + p + ' img2/' + _filename + '/' + p)
    print 'done'


if __name__ == '__main__':
    os.system('mkdir img2')
    main()
