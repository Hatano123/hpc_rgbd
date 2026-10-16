"""RGB baseline compatibility CLI."""
import sys
from rgbd_yolo.cli import main

if __name__ == '__main__':
    main(['train', '--mode', 'rgb', *sys.argv[1:]])
