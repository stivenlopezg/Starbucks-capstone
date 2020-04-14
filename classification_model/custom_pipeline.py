import pandas as pd
from datetime import date
from sklearn.base import TransformerMixin, BaseEstimator
from classification_model.config import gender_categories, offer_categories, channel_categories


class CalculateAntiquity(BaseEstimator, TransformerMixin):
    def __init__(self, column: str):
        if not isinstance(column, str):
            self.column = str(column)
        else:
            self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        X['antiquity'] = -(X[self.column] - pd.to_datetime(date.today().strftime('%Y-%m-%d'))).dt.days
        return X


class NumberChannels(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        X['number_channels'] = X[self.columns].sum(axis=1)
        return X


class Proportion(BaseEstimator, TransformerMixin):
    def __init__(self, numerator: str, denominator: str):
        if not isinstance(numerator, str):
            self.numerator = str(numerator)
        else:
            self.numerator = numerator
        if not isinstance(denominator, str):
            self.denominator = str(denominator)
        else:
            self.denominator = denominator

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        X['reward/difficulty'] = X[self.numerator] / X[self.denominator]
        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.loc[:, self.columns]
        return X


class ConvertDtypes(BaseEstimator, TransformerMixin):
    def __init__(self, numerical: list, categorical: list, date: list):
        if not isinstance(numerical, list):
            self.numerical = [numerical]
        else:
            self.numerical = numerical
        if not isinstance(categorical, list):
            self.categorical = [categorical]
        else:
            self.categorical = categorical
        if not isinstance(date, list):
            self.date = [date]
        else:
            self.date = date

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for numerical in self.numerical:
            X[numerical] = pd.to_numeric(X[numerical])
        for categorical in self.categorical:
            if categorical == 'gender':
                categories = gender_categories
            elif categorical == 'offer_type':
                categories = offer_categories
            else:
                categories = channel_categories
            X[categorical] = pd.Categorical(X[categorical], categories=categories)
        for date in self.date:
            X[date] = pd.to_datetime(X[date], format='%Y%m%d')
        return X


class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.get_dummies(X, columns=self.columns)
        return X


class GetDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.columns)
        return X
