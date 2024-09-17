import streamlit as st 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

def load_model(filename):
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        st.stop()
    except ValueError as e:
        st.error(f"ValueError during model loading: {e}")
        st.text(traceback.format_exc())
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.text(traceback.format_exc())
        st.stop()


loaded_model = load_model('final_model.pkl')

st.title("Customer Segmentation")

@st.cache_data
def predict_cluster(data):
    try:
        return loaded_model.predict(data)[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.text(traceback.format_exc())
        st.stop()

with st.form("my_form"):


    balance = st.number_input('Balance', step=0.001, format="%.6f", min_value=0.0, max_value=1000000.0, help="The total amount of money available in the account.")
    balance_frequency = st.number_input('Balance Frequency', step=0.001, format="%.6f", min_value=0.0, max_value=1.0, help="How often the account balance is updated (0 to 1 scale).")
    purchases = st.number_input('Purchases', step=0.01, format="%.2f", min_value=0.0, max_value=50000.0, help="Total amount spent on purchases.")
    oneoff_purchases = st.number_input('OneOff Purchases', step=0.01, format="%.2f", min_value=0.0, max_value=10000.0, help="Amount spent on one-off purchases.")
    installments_purchases = st.number_input('Installments Purchases', step=0.01, format="%.2f", min_value=0.0, max_value=10000.0, help="Amount spent on installment purchases.")
    cash_advance = st.number_input('Cash Advance', step=0.01, format="%.6f", min_value=0.0, max_value=10000.0, help="Amount of cash advances taken from the account.")
    purchases_frequency = st.number_input('Purchases Frequency', step=0.01, format="%.6f", min_value=0.0, max_value=1.0, help="How frequently purchases are made (0 to 1 scale).")
    oneoff_purchases_frequency = st.number_input('OneOff Purchases Frequency', step=0.1, format="%.6f", min_value=0.0, max_value=1.0, help="Frequency of one-off purchases (0 to 1 scale).")
    purchases_installment_frequency = st.number_input('Purchases Installments Frequency', step=0.1, format="%.6f", min_value=0.0, max_value=1.0, help="Frequency of installment purchases (0 to 1 scale).")
    cash_advance_frequency = st.number_input('Cash Advance Frequency', step=0.1, format="%.6f", min_value=0.0, max_value=1.0, help="Frequency of cash advances (0 to 1 scale).")
    cash_advance_trx = st.number_input('Cash Advance Trx', step=1, min_value=0, max_value=100, help="Number of cash advance transactions.")
    purchases_trx = st.number_input('Purchases TRX', step=1, min_value=0, max_value=500, help="Number of purchase transactions.")
    credit_limit = st.number_input('Credit Limit', step=0.1, format="%.1f", min_value=0.0, max_value=1000000.0, help="Maximum credit limit on the account.")
    payments = st.number_input('Payments', step=0.01, format="%.6f", min_value=0.0, max_value=50000.0, help="Total amount of payments made on the account.")
    minimum_payments = st.number_input('Minimum Payments', step=0.01, format="%.6f", min_value=0.0, max_value=10000.0, help="Minimum payments made on the account.")
    prc_full_payment = st.number_input('PRC Full Payment', step=0.01, format="%.6f", min_value=0.0, max_value=1.0, help="Percentage of full payments made (0 to 1 scale).")
    tenure = st.number_input('Tenure', step=1, min_value=0, max_value=120, help="Number of months the account has been active.")


    data = pd.DataFrame([[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
                          purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency, cash_advance_frequency,
                          cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]],
                        columns=['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
                                 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
                                 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX',
                                 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'])

    submitted = st.form_submit_button("Submit")

if submitted:
    try:
        clust = predict_cluster(data)
        st.write('Data Belongs to Cluster', clust)

        
        df = pd.read_csv('Clustered_Customer_Data.csv')  

    
        df.columns = [col.upper() for col in df.columns]

        cluster_df1 = df[df['CLUSTER'] == clust]
        plt.rcParams["figure.figsize"] = (20, 3)
        for c in cluster_df1.drop(['CLUSTER'], axis=1):
            fig, ax = plt.subplots()
            sns.histplot(cluster_df1[c], ax=ax, kde=True)
            ax.set_title(f"Distribution of {c} in Cluster {clust}")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during prediction or visualization: {e}")
        st.text(traceback.format_exc())
