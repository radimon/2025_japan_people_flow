from pathlib import Path
import pandas as pd


def build_density_table(csv_path: str | Path) -> pd.DataFrame:
    """
    Build spatiotemporal density table from individual-level mobility data.

    Expected input columns:
        uid : individual identifier
        d   : day index
        t   : time slot
        x   : spatial grid x
        y   : spatial grid y

    Output columns:
        d, t, x, y, count (number of unique individuals)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    expected = {"uid", "d", "t", "x", "y"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    for c in ["uid", "d", "t", "x", "y"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["uid", "d", "t", "x", "y"])

    density = (
        df.groupby(["d", "t", "x", "y"])["uid"]
          .nunique()
          .reset_index(name="count")
          .sort_values(["d", "t", "count"], ascending=[True, True, False])
    )

    return density


def save_density(density: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    density.to_parquet(out_path, index=False)
