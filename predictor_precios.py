import pandas as pd
import numpy as np
import xgboost as xgb

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.xgboost import XgboostModelArtifact

@env(infer_pip_packages=True)
# @env(pip_packages=['xgboost'])

@artifacts(XgboostModelArtifact('model'))

class PredictorPrecios(BentoService):
    """
    A minimum prediction service exposing a XGB model
    """

    @api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        dmatrix = xgb.DMatrix(df)
        return self.artifacts.model.predict(dmatrix)