"""Generate final top-k ranking — delegates to retention rank."""
import sys
sys.argv = ["retention", "rank"]
from retention.cli import main
main()
