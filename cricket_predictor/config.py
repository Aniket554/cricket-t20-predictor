"""Paths and constants for the T20 predictor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw_data" / "cricsheet"
DATA_DIR = ROOT / "data"

T20_TYPES = {"T20", "IT20"}

# Cricsheet over index is 0-based: overs 1-6 -> 0..5
PHASES = ("powerplay", "middle", "death")


def phase_of(over: int) -> str:
    if over < 6:
        return "powerplay"
    if over < 15:
        return "middle"
    return "death"


# Dismissals that are NOT credited to the bowler
NON_BOWLER_KINDS = {"run out", "retired hurt", "retired out", "obstructing the field"}

# "Recent form" = previous N innings (batters) / bowling spells (bowlers)
BASELINE_WINDOW = 20
MIN_PRIOR = 5            # need this many prior appearances to be modelled
DEFAULT_PAR = 155.0      # fallback first-innings par for unseen venues

# Competition zips on cricsheet.org/downloads (verify names there; they occasionally change)
DEFAULT_ZIPS = ["ipl_json", "bbl_json", "psl_json", "cpl_json", "sat_json",
                "ilt_json", "mlc_json", "ntb_json", "lpl_json", "bpl_json", "t20s_json"]

# Smoother target: average over this player's next N appearances (current one included)
FORWARD_WINDOW = 5
