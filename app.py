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

st.set_page_config(page_title="Customer Segmentation System", page_icon=":bar_chart:")

st.markdown("""
    <style>
        .header {
            text-align: center;
            font-size: 36px;
            color: #4CAF50;
            margin-bottom: 20px;
        }
        .description {
            font-size: 18px;
            color: #FFFFFF; /* White color for better readability on black background */
            text-align: center;
            margin-bottom: 30px;
        }
        .widget {
            margin: 10px 0;
        }
        }
    </style>
    <div class="header">Welcome to the Customer Segmentation Application!</div>
    <div class="description">
        This tool classifies customers into different segments based on their credit card information.
        Fill in the details below to see which customer cluster you belong to and gain insights into your financial profile.
    </div>
""", unsafe_allow_html=True)

@st.cache_data
def predict_cluster(data):
    try:
        return loaded_model.predict(data)[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.text(traceback.format_exc())
        st.stop()

with st.form("my_form"):
    st.subheader("Input Customer Information")

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
    prc_full_payment = st.number_input('PRC Full Payment', step=0.01, format="%.6f", min_value=0.0, max_value=1.0000, help="Percentage of full payments made (0 to 1 scale).")
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
        st.write('### Data Belongs to Cluster', clust)

        df = pd.read_csv('Clustered_Customer_Data.csv')
        df.columns = [col.upper() for col in df.columns]

        cluster_df1 = df[df['CLUSTER'] == clust]
        st.write("#### Cluster Data Overview")
        st.write(f"**Number of records in this cluster:** {cluster_df1.shape[0]}")

        plt.rcParams["figure.figsize"] = (20, 3)
        for c in cluster_df1.drop(['CLUSTER'], axis=1):
            fig, ax = plt.subplots()
            sns.histplot(cluster_df1[c], ax=ax, kde=True)
            ax.set_title(f"Distribution of {c} in Cluster {clust}")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during prediction or visualization: {e}")
        st.text(traceback.format_exc())
# import streamlit as st
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
# import traceback
# import io

# def load_model(filename):
#     try:
#         with open(filename, 'rb') as f:
#             model = pickle.load(f)
#         return model
#     except FileNotFoundError:
#         st.error("Model file not found. Please check the file path.")
#         st.stop()
#     except ValueError as e:
#         st.error(f"ValueError during model loading: {e}")
#         st.text(traceback.format_exc())
#         st.stop()
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         st.text(traceback.format_exc())
#         st.stop()

# loaded_model = load_model('final_model.pkl')

# st.set_page_config(page_title="Customer Segmentation System", page_icon=":bar_chart:")

# st.markdown("""
#     <style>
#         .header {
#             text-align: center;
#             font-size: 36px;
#             color: #4CAF50;
#             margin-bottom: 20px;
#         }
#         .description {
#             font-size: 18px;
#             color: #FFFFFF; /* White color for better readability on black background */
#             text-align: center;
#             margin-bottom: 30px;
#         }
#         .widget {
#             margin: 10px 0;
#         }
#     </style>
#     <div class="header">Welcome to the Customer Segmentation Application!</div>
#     <div class="description">
#         This tool classifies customers into different segments based on their credit card information.
#         Fill in the details below to see which customer cluster you belong to and gain insights into your financial profile.
#     </div>
# """, unsafe_allow_html=True)

# @st.cache_data
# def predict_cluster(data):
#     try:
#         return loaded_model.predict(data)[0]
#     except Exception as e:
#         st.error(f"Error during prediction: {e}")
#         st.text(traceback.format_exc())
#         st.stop()

# @st.cache_data
# def get_feature_importances():
#     try:
#         # Get feature importances from the model
#         feature_importances = loaded_model.feature_importances_
#         return feature_importances
#     except Exception as e:
#         st.error(f"Error extracting feature importances: {e}")
#         st.text(traceback.format_exc())
#         st.stop()

# with st.form("my_form"):
#     balance = st.number_input('Balance', step=0.001, format="%.6f", help="The total amount of money available in the account.")
#     balance_frequency = st.number_input('Balance Frequency', step=0.001, format="%.6f", help="How often the account balance is updated.")
#     purchases = st.number_input('Purchases', step=0.01, format="%.2f", help="Total amount spent on purchases.")
#     oneoff_purchases = st.number_input('OneOff Purchases', step=0.01, format="%.2f", help="Amount spent on one-off purchases.")
#     installments_purchases = st.number_input('Installments Purchases', step=0.01, format="%.2f", help="Amount spent on installment purchases.")
#     cash_advance = st.number_input('Cash Advance', step=0.01, format="%.6f", help="Amount of cash advances taken from the account.")
#     purchases_frequency = st.number_input('Purchases Frequency', step=0.01, format="%.6f", help="How frequently purchases are made.")
#     oneoff_purchases_frequency = st.number_input('OneOff Purchases Frequency', step=0.1, format="%.6f", help="Frequency of one-off purchases.")
#     purchases_installment_frequency = st.number_input('Purchases Installments Frequency', step=0.1, format="%.6f", help="Frequency of installment purchases.")
#     cash_advance_frequency = st.number_input('Cash Advance Frequency', step=0.1, format="%.6f", help="Frequency of cash advances.")
#     cash_advance_trx = st.number_input('Cash Advance Trx', step=1, help="Number of cash advance transactions.")
#     purchases_trx = st.number_input('Purchases TRX', step=1, help="Number of purchase transactions.")
#     credit_limit = st.number_input('Credit Limit', step=0.1, format="%.1f", help="Maximum credit limit on the account.")
#     payments = st.number_input('Payments', step=0.01, format="%.6f", help="Total amount of payments made on the account.")
#     minimum_payments = st.number_input('Minimum Payments', step=0.01, format="%.6f", help="Minimum payments made on the account.")
#     prc_full_payment = st.number_input('PRC Full Payment', step=0.01, format="%.6f", help="Percentage of full payments made.")
#     tenure = st.number_input('Tenure', step=1, help="Number of months the account has been active.")
   

#     data = pd.DataFrame([[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
#                           purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency, cash_advance_frequency,
#                           cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]],
#                         columns=['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
#                                  'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
#                                  'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX',
#                                  'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'])

#     submitted = st.form_submit_button("Submit")


# import io

# def generate_report(cluster_df):
#     buffer = io.BytesIO()
#     cluster_df.describe().to_csv(buffer)
#     buffer.seek(0)
#     return buffer.getvalue()

# # Inside the Streamlit app
# if submitted:
#     try:
#         clust = predict_cluster(data)
#         st.write('### Data Belongs to Cluster', clust)

#         df = pd.read_csv('Clustered_Customer_Data.csv')
#         df.columns = [col.upper() for col in df.columns]

#         cluster_df1 = df[df['CLUSTER'] == clust]
#         st.write("#### Cluster Data Overview")
#         st.write(f"**Number of records in this cluster:** {cluster_df1.shape[0]}")

#         st.write("#### Summary Statistics for Cluster", clust)
#         cluster_summary = cluster_df1.describe().T
#         st.write(cluster_summary)

#         # Extract feature importances
#         feature_importances = get_feature_importances()
#         feature_names = df.columns[:-1]  # Assuming the last column is 'CLUSTER'

#         if len(feature_importances) == len(feature_names):
#             st.write("#### Feature Importance")
#             feature_importances_series = pd.Series(feature_importances, index=feature_names)
#             st.bar_chart(feature_importances_series)
#         else:
#             st.warning("The number of feature importances does not match the number of features in the data.")

#         st.write("#### Scatter Plot of Key Features")
#         fig, ax = plt.subplots()
#         sns.scatterplot(x='BALANCE', y='PURCHASES', data=cluster_df1, ax=ax)
#         ax.set_title('Balance vs Purchases in Cluster ' + str(clust))
#         st.pyplot(fig)

#         st.write("#### Interactive Data Table")
#         st.dataframe(cluster_df1)

#         # Add useful insights
#         st.write("#### Key Insights for Your Cluster")
#         st.write("""
#             **Percentage of Returning Loans**: The percentage of customers in this cluster who return their loans on time.
#         """)
        
#         returning_loans_percentage = cluster_df1['PRC_FULL_PAYMENT'].mean() * 100
#         st.markdown(f"**Returning Loans Percentage:** {returning_loans_percentage:.2f}%")

#         st.write("#### Download Report")
#         report = generate_report(cluster_df1)
#         st.download_button("Download Report", data=report, file_name="cluster_report.csv", mime="text/csv")

#     except Exception as e:
#         st.error(f"Error during prediction or visualization: {e}")
#         st.text(traceback.format_exc())
