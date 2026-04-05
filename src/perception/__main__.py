"""
Entry point for running perception as a module.
Usage: python -m perception <image_path>
"""

import logging
import sys

from perception.main import main

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python -m perception <image_path> [--output <output_path>]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = None
    
    if "--output" in sys.argv:
        output_idx = sys.argv.index("--output")
        if output_idx + 1 < len(sys.argv):
            output_path = sys.argv[output_idx + 1]
    
    main(image_path, output_path)
