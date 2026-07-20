"""Build champion bundle — delegates to retention bundle."""
import sys
sys.argv = ["retention", "bundle"]
from retention.cli import main
main()
