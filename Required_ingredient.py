import pandas as pd

df = pd.read_excel(r"C:\Users\ASUS\Downloads\Pizza_ingredients.xlsx")

df1 = pd.read_excel("pred_out.xlsx")

merged_df = pd.merge(df, df1, on='pizza_name_id', how='inner')

merged_df['Total_requirement'] = merged_df['Items_Qty_In_Grams'] * merged_df['Predicted Quantity for Week 54']

final_requirement = merged_df.groupby(['pizza_ingredients'])['Total_requirement'].sum().reset_index()

final_requirement.to_excel("Required ingredients for next week")