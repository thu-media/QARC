import os
TEST_FOLDER = './mov'

def main():
    test_files = os.listdir(TEST_FOLDER)
    for files in test_files:
        os.system('python generate.py ' + files + '&')

if __name__ == '__main__':
    main()