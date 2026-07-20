"""Train all challengers — delegates to retention train."""
import sys
sys.argv = ["retention", "train"]
from retention.cli import main
main()
