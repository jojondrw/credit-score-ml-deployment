import streamlit as st
import pandas as pd
from inference import CreditScoringInference

# Load model + preprocessing 
infer = CreditScoringInference()

# Opsi dropdown (nilai persis dari dataset)
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August']
OCCUPATIONS = ['Journalist', 'Teacher', 'Developer', 'Architect', 'Doctor', 'Media_Manager',
               'Accountant', 'Entrepreneur', 'Lawyer', 'Writer', 'Engineer', 'Manager',
               'Musician', 'Scientist', 'Mechanic']
BEHAVIOURS = ['High_spent_Small_value_payments', 'Low_spent_Medium_value_payments',
              'Low_spent_Large_value_payments', 'Low_spent_Small_value_payments',
              'High_spent_Medium_value_payments', 'High_spent_Large_value_payments']
LOANS = ['Auto_Loan', 'Credit_Builder_Loan', 'Debt_Consolidation_Loan', 'Home_Equity_Loan',
         'Mortgage_Loan', 'Payday_Loan', 'Personal_Loan', 'Student_Loan']


def main():
    st.set_page_config(page_title="Credit Score Prediction", layout="wide")
    st.title("Credit Score Prediction")

    with st.form("input_form"):
        st.subheader("Profil & Pendapatan")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 0, 100, 30)
            occupation = st.selectbox("Occupation", OCCUPATIONS)
            month = st.selectbox("Month", MONTHS)
        with c2:
            annual_income = st.number_input("Annual Income", 0.0, value=20000.0)
            monthly_salary = st.number_input("Monthly Inhand Salary", 0.0, value=1600.0)
            monthly_balance = st.number_input("Monthly Balance", 0.0, value=300.0)
        with c3:
            amount_invested = st.number_input("Amount Invested Monthly", 0.0, value=90.0)
            total_emi = st.number_input("Total EMI per Month", 0.0, value=30.0)
            credit_util = st.number_input("Credit Utilization Ratio", 0.0, value=30.0)

        st.subheader("Kredit & Pinjaman")
        c4, c5, c6 = st.columns(3)
        with c4:
            num_bank = st.number_input("Num Bank Accounts", 0, value=5)
            num_card = st.number_input("Num Credit Card", 0, value=5)
            num_loan = st.number_input("Num of Loan", 0, value=3)
        with c5:
            interest = st.number_input("Interest Rate", 0, 100, 15)
            outstanding = st.number_input("Outstanding Debt", 0.0, value=1000.0)
            changed_limit = st.number_input("Changed Credit Limit", 0.0, value=10.0)
        with c6:
            credit_mix = st.selectbox("Credit Mix", ['Bad', 'Standard', 'Good'])
            credit_inquiries = st.number_input("Num Credit Inquiries", 0, value=5)

        st.subheader("Riwayat Pembayaran")
        c7, c8, c9 = st.columns(3)
        with c7:
            delay_due = st.number_input("Delay from Due Date", 0, value=20)
            num_delayed = st.number_input("Num of Delayed Payment", 0, value=10)
        with c8:
            credit_history = st.number_input("Credit History Age (bulan)", 0, value=200)
            pay_min = st.radio("Payment of Min Amount", ['Yes', 'No'])
        with c9:
            pay_behaviour = st.selectbox("Payment Behaviour", BEHAVIOURS)

        st.subheader("Jenis Pinjaman (centang yang dimiliki)")
        loan_cols = st.columns(4)
        loan_values = {}
        for i, loan in enumerate(LOANS):
            with loan_cols[i % 4]:
                loan_values[loan] = int(st.checkbox(loan.replace('_', ' ')))

        submit = st.form_submit_button("Predict")

    if submit:
        data = {
            'Age': age, 'Annual_Income': annual_income, 'Monthly_Inhand_Salary': monthly_salary,
            'Num_Bank_Accounts': num_bank, 'Num_Credit_Card': num_card, 'Interest_Rate': interest,
            'Num_of_Loan': num_loan, 'Delay_from_due_date': delay_due,
            'Num_of_Delayed_Payment': num_delayed, 'Changed_Credit_Limit': changed_limit,
            'Num_Credit_Inquiries': credit_inquiries, 'Outstanding_Debt': outstanding,
            'Credit_Utilization_Ratio': credit_util, 'Credit_History_Age': credit_history,
            'Total_EMI_per_month': total_emi, 'Amount_invested_monthly': amount_invested,
            'Monthly_Balance': monthly_balance, 'Credit_Mix': credit_mix, 'Month': month,
            'Occupation': occupation, 'Payment_of_Min_Amount': pay_min,
            'Payment_Behaviour': pay_behaviour
        }
        data.update(loan_values)   # tambah 8 kolom pinjaman

        result = infer.predict(data)

        st.subheader("Hasil Prediksi")
        score = result['prediction']
        if score == 'Good':
            st.success(f"Credit Score: {score}")
        elif score == 'Standard':
            st.info(f"Credit Score: {score}")
        else:
            st.error(f"Credit Score: {score}")
        st.write("Probabilitas tiap kelas:")
        st.bar_chart(pd.Series(result['probabilities']))


if __name__ == "__main__":
    main()