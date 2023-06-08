
########################################################################
# ASSOCATING RULES PROJECT FOR ARMUT COMPANY
########################################################################

####################
#1- Preapering DATA
####################

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
import datetime as dt
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


df_ = pd.read_csv("armut_data.csv")
df = df_.copy()
df.head()
df.info()
df.isnull().sum()

######################
# Removal of Association Rules
######################

df["Service"] = df["ServiceId"].astype(str) + str("_") + df["CategoryId"].astype(str)

df["CreateDate"] = pd.to_datetime(df["CreateDate"])

df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")

df["Cart_id"] = df["UserId"].astype(str) + str("_") + df["New_Date"].astype(str)



df_car_ser = df.groupby(['Cart_id', 'Service']).size().unstack(fill_value=0)
df_car_ser = df_car_ser.applymap(lambda x:1 if x>0 else 0)



frequent_services = apriori(df_car_ser,
                            min_support=0.01,
                            use_colnames=True)

frequent_services.sort_values("support", ascending=False)

rules = association_rules(frequent_services,
                          metric="support",
                          min_threshold=0.01)


rules.sort_values("confidence", ascending=False)

sorted_rules = rules.sort_values("lift",ascending=False)

##################################
# Creating arl_recommender function
##################################
########
#sample
########
service_id = str("2_0")

recommendation_list=[]

for i, service in enumerate(sorted_rules["antecedents"]):
    for j in list(service):
        if j == service_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

#########
#Function
#########

def arl_recommender(rules_df, service_id):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, service in sorted_rules["antecedents"].iteritems():
        if service_id in list(service):
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list


arl_recommender(rules, "2_0")
arl_recommender(rules, "9_4")
arl_recommender(rules, "15_1")
