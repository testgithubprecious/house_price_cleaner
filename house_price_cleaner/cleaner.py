import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel

class HousePricePreprocessor:
    def __init__(self, ordinal_mappings, ordinal_features):
        self.ordinal_mappings = ordinal_mappings
        self.ordinal_features = ordinal_features
        self.low_card_columns = []
        self.high_card_columns = []
        self.low_card_dummies = None
        self.target_encoder = None
        self.discrete = []
        self.continuous = []
        self.scaler = MinMaxScaler()
        self.discrete_ohe_columns = []
        self.fitted = False
        self.dummy_template = None
        self.selector = None  # 🔹 Feature selector
        self.feature_names = []  # 🔹 Keep original feature names

    def _generate_dummy_template(self, train_df, test_df):
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df['is_train'] = 1
        test_df['is_train'] = 0
        test_df['SalePrice'] = 0
        combined = pd.concat([train_df, test_df], axis=0)
        combined['MSSubClass'] = combined['MSSubClass'].astype(str)
        combined['MoSold'] = combined['MoSold'].astype(str)
        combined['YrSold'] = combined['YrSold'].astype(str)
        combined = pd.get_dummies(combined, columns=['MSSubClass', 'MoSold', 'YrSold'], drop_first=True)
        self.dummy_template = combined.drop(['SalePrice', 'is_train'], axis=1).columns.tolist()
        train_encoded = combined[combined['is_train'] == 1].drop(['is_train'], axis=1)
        test_encoded = combined[combined['is_train'] == 0].drop(['is_train', 'SalePrice'], axis=1)
        return train_encoded, test_encoded

    def fill_numerical_missing(self, df):
        df = df.copy()
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
        df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].median())
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
        return df

    def map_ordinal(self, df):
        df = df.copy()
        for col, mapping in self.ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)
        return df

    def separate_categorical(self, df):
        categorical = df.select_dtypes(include='object')
        self.low_card_columns = [col for col in categorical.columns if df[col].nunique() <= 5]
        self.high_card_columns = [col for col in categorical.columns if df[col].nunique() > 5]

    def encode_low_card(self, df):
        df = df.copy()
        df[self.low_card_columns] = df[self.low_card_columns].fillna("Missing")
        dummies = pd.get_dummies(df[self.low_card_columns], drop_first=True)
        self.low_card_dummies = dummies.columns.tolist()
        return dummies

    def encode_high_card(self, df, y=None, fit=True):
        df = df.copy()
        if fit:
            self.target_encoder = ce.TargetEncoder(cols=self.high_card_columns)
            encoded = self.target_encoder.fit_transform(df[self.high_card_columns], y)
        else:
            encoded = self.target_encoder.transform(df[self.high_card_columns])
        return encoded

    def preprocess_discrete(self, df, fit=True):
        df = df.copy()
        df['HasLowQualFinSF'] = df['LowQualFinSF'].apply(lambda x: 1 if x > 0 else 0)
        df['Has3SsnPorch'] = df['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)
        df['HasMiscVal'] = df['MiscVal'].apply(lambda x: 1 if x > 0 else 0)
        df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        df.drop(['LowQualFinSF', '3SsnPorch', 'MiscVal', 'PoolArea'], axis=1, inplace=True)
        df['BsmtHalfBath'] = df['BsmtHalfBath'].apply(lambda x: 1 if x > 1 else x)
        df['HalfBath'] = df['HalfBath'].apply(lambda x: 1 if x > 1 else x)
        df['MSSubClass'] = df['MSSubClass'].astype(str)
        df['MoSold'] = df['MoSold'].astype(str)
        df['YrSold'] = df['YrSold'].astype(str)
        df = pd.get_dummies(df, columns=['MSSubClass', 'MoSold', 'YrSold'], drop_first=True)
        if fit:
            self.discrete_ohe_columns = df.columns.tolist()
        else:
            for col in self.discrete_ohe_columns:
                if col not in df:
                    df[col] = 0
            df = df[self.discrete_ohe_columns]
        return df

    def process_continuous(self, df, fit_scaler=True):
        df = df.copy()
        numeric_df = df[self.continuous].select_dtypes(include=["int64", "float64"])
        df_transformed = numeric_df.apply(lambda x: np.log1p(x))
        if fit_scaler:
            scaled = pd.DataFrame(self.scaler.fit_transform(df_transformed), columns=df_transformed.columns)
        else:
            scaled = pd.DataFrame(self.scaler.transform(df_transformed), columns=df_transformed.columns)
        return scaled

    def fit_transform(self, df, y, test_df=None):
        df = self.fill_numerical_missing(df)
        df = self.map_ordinal(df)
        self.separate_categorical(df)
        low_encoded = self.encode_low_card(df)
        high_encoded = self.encode_high_card(df, y, fit=True)
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        self.discrete = [col for col in numeric_df.columns if numeric_df[col].nunique() < 25 and numeric_df[col].dtype == 'int64']
        self.continuous = [col for col in numeric_df.columns if col not in self.discrete and col != 'SalePrice']
        discrete_df = self.preprocess_discrete(numeric_df[self.discrete], fit=True)
        continuous_df = self.process_continuous(df, fit_scaler=True)
        final = pd.concat([continuous_df, discrete_df, low_encoded, high_encoded], axis=1)

        # 🔹 Feature selection
        self.feature_names = final.columns.tolist()
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
        xgb.fit(final, y)
        self.selector = SelectFromModel(xgb, prefit=True, threshold='median')
        selected = self.selector.transform(final)
        selected_df = pd.DataFrame(selected, columns=np.array(self.feature_names)[self.selector.get_support()])
        self.fitted = True

        if test_df is not None:
            _, _ = self._generate_dummy_template(numeric_df[self.discrete], test_df[self.discrete])

        return selected_df

    def transform(self, df):
        if not self.fitted:
            raise Exception("You must call fit_transform first.")
        df = self.fill_numerical_missing(df)
        df = self.map_ordinal(df)
        low_encoded = pd.get_dummies(df[self.low_card_columns].fillna("Missing"), drop_first=True)
        for col in self.low_card_dummies:
            if col not in low_encoded:
                low_encoded[col] = 0
        low_encoded = low_encoded[self.low_card_dummies]
        high_encoded = self.encode_high_card(df, fit=False)
        discrete_df = self.preprocess_discrete(df[self.discrete], fit=False)
        for col in self.discrete_ohe_columns:
            if col not in discrete_df:
                discrete_df[col] = 0
        discrete_df = discrete_df[self.discrete_ohe_columns]
        continuous_df = self.process_continuous(df, fit_scaler=False)
        final = pd.concat([continuous_df, discrete_df, low_encoded, high_encoded], axis=1)

        # 🔹 Apply saved selector
        selected = self.selector.transform(final)
        selected_df = pd.DataFrame(selected, columns=np.array(self.feature_names)[self.selector.get_support()])
        return selected_df

