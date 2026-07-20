"""Create dev/test split — delegates to retention split."""
import sys
sys.argv = ["retention", "split"]
from retention.cli import main
main()
