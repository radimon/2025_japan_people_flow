import numpy as np
import pandas as pd


class GridLatLngMapper:
    """
    Approximate mapping from grid coordinates (x, y)
    to geographic coordinates (lat, lng)
    using anchor-based affine approximation.
    """

    def __init__(self, anchors):
        self.anchors = pd.DataFrame(anchors)
        self._fit()

    def _fit(self):
        # lat = a*x + b*y + c
        A = np.column_stack([
            self.anchors["x"],
            self.anchors["y"],
            np.ones(len(self.anchors))
        ])

        self.lat_coef, *_ = np.linalg.lstsq(A, self.anchors["lat"], rcond=None)
        self.lng_coef, *_ = np.linalg.lstsq(A, self.anchors["lng"], rcond=None)

    def transform(self, df, x_col="x", y_col="y"):
        A = np.column_stack([
            df[x_col],
            df[y_col],
            np.ones(len(df))
        ])

        df = df.copy()
        df["lat"] = A @ self.lat_coef
        df["lng"] = A @ self.lng_coef
        return df
