"""Compatibility CLI. Use `python -m rgbd_yolo train` for new workflows."""
import sys
from rgbd_yolo.cli import main

if __name__ == '__main__':
    main(['train', *sys.argv[1:]])
