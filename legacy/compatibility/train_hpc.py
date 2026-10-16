"""Compatibility CLI for the consolidated training implementation."""
import sys
from rgbd_yolo.cli import main

if __name__ == '__main__':
    main(['train', *sys.argv[1:]])
