import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


class Preprocessing:
    def __init__(self):
        # Kolom yang tidak ada nilai prediktif
        self.cols_to_drop = ['Unnamed: 0', 'ID', 'Customer_ID', 'SSN', 'Name']

        # String kotor diganti jadi NaN
        self.dirty_values = ['_______', '_', 'NM', '!@9#%8']

        # Kolom numerik yang kebaca object
        self.num_obj_cols = ['Age', 'Annual_Income', 'Num_of_Loan', 'Changed_Credit_Limit',
                             'Num_of_Delayed_Payment', 'Outstanding_Debt', 'Amount_invested_monthly']

        # Kolom yang tidak mungkin negatif
        self.invalid_negatives = ['Num_of_Loan', 'Num_Bank_Accounts', 'Delay_from_due_date', 'Age', 'Num_of_Delayed_Payment']

        # Kolom dengan nilai maks tidak mungkin
        self.invalid_max = {'Age': 100, 'Num_Bank_Accounts': 20, 'Num_Credit_Card': 20, 'Num_of_Loan': 15, 'Num_of_Delayed_Payment': 30, 'Num_Credit_Inquiries': 20}

        # Jenis pinjaman untuk multi-hot Type_of_Loan ("Not Specified" diabaikan)
        self.loan_types = ['Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan',
                           'Home Equity Loan', 'Mortgage Loan', 'Payday Loan',
                           'Personal Loan', 'Student Loan']

        # Define features untuk ColumnTransformer
        self.num_features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                             'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                             'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                             'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
                             'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
                             'Auto_Loan', 'Credit_Builder_Loan', 'Debt_Consolidation_Loan',
                             'Home_Equity_Loan', 'Mortgage_Loan', 'Payday_Loan',
                             'Personal_Loan', 'Student_Loan']
        
        self.ordinal_features = ['Credit_Mix']

        self.cat_features = ['Month', 'Occupation', 'Payment_of_Min_Amount', 'Payment_Behaviour']

        # Preprocessing punya (HAS-A) preprocessor sebagai attribute
        self.preprocessor = self.build_preprocessor()

    def build_preprocessor(self):
        # numerik: median + scaling
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # ordinal: Credit_Mix punya urutan natural
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=[['Bad', 'Standard', 'Good']]))
        ])

        # categorical: tidak ada urutan, pakai OHE
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.num_features),
            ('ord', ordinal_transformer, self.ordinal_features),
            ('cat', categorical_transformer, self.cat_features)
        ])
        return preprocessor

    def clean(self, df):
        df = df.copy()

        # 1. Drop indentifiers
        df = df.drop(columns=self.cols_to_drop, errors='ignore')

        # 2. Ganti jadi NaN
        df = df.replace(self.dirty_values, np.nan)

        # 3. Ubah object jadi numerical (float)
        for col in self.num_obj_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                df[col] = df[col].replace('', np.nan).astype(float)

        # 4. Nilai negatif yang tidak masuk akal jadi NaN
        for col in self.invalid_negatives:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan

        # 5. Nilai max yang tidak masuk akal jadi NaN
        for col, threshold in self.invalid_max.items():
            df.loc[df[col] > threshold, col] = np.nan
        
        # 6. Credit_History_Age jadi total bulan
        ekstrak = df['Credit_History_Age'].str.extract(r'(\d+) Years and (\d+) Months')
        df['Credit_History_Age'] = (ekstrak[0].astype(float) * 12) + ekstrak[1].astype(float)

        # 7. Cap Interest_Rate di 100
        df['Interest_Rate'] = df['Interest_Rate'].clip(upper=100)

        # 8. Multi-hot Type_of_Loan jadi 8 kolom biner per jenis ("Not Specified" diabaikan, NaN otomatis jadi 0 di semua kolom)
        for loan in self.loan_types:
            col = loan.replace(' ', '_').replace('-', '_')
            df[col] = df['Type_of_Loan'].fillna('').str.contains(loan, regex=False).astype('int64')

        df = df.drop(columns=['Type_of_Loan'], errors='ignore')

        return df

    def fit_transform(self, X):
        return self.preprocessor.fit_transform(X)

    def transform(self, X):
        return self.preprocessor.transform(X)
