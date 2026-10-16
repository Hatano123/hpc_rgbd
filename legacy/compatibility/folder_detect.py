"""Compatibility CLI; explicitly supply --model, --image and --output."""
import sys
from rgbd_yolo.cli import main

if __name__ == '__main__':
    main(['predict', *sys.argv[1:]])
