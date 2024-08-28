import pandas as pd
import numpy as np
import joblib 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

file_path = '16_1-2.csv'
df = pd.read_csv(file_path) #Adjust with the csv you want to clean
file_name = os.path.splitext(os.path.basename(file_path))[0]

# Some of the logs data have the first row set to units value, so adjust to the well logs to remove them
# df.iloc[1:].reset_index(drop=True) 
df = df.apply(pd.to_numeric, errors='coerce')

columnscheck = ['GR', 'RHOB', 'NPHI', 'DTC', 'RDEP', 'RXO', 'RSHA', 'FORCE_2020_LITHOFACIES_LITHOLOGY']
for column in columnscheck:
    if column not in df.columns:
        df[column] = np.nan

columnsupdate = ['GR', 'RHOB', 'NPHI', 'DTC', 'RDEP', 'RXO', 'RSHA']
for column in columnsupdate:
    df.loc[(df[column] < -200.0) & (df[column] > -999.9999), column] = np.nan
        
filtered_model = ['GR', 'RHOB', 'NPHI', 'DTC', 'RDEP', 'RXO', 'RSHA', 'FORCE_2020_LITHOFACIES_LITHOLOGY']
df_filtered_model = df[filtered_model]
other_columns = df.drop(columns=filtered_model)
df_filtered_model_imputed = df_filtered_model.copy()

models = {
    'GR': joblib.load('GR_regressor.pkl'),
    'RHOB': joblib.load('RHOB_regressor.pkl'),
    'NPHI': joblib.load('NPHI_regressor.pkl'),
    'DTC': joblib.load('DTC_regressor.pkl'),
    'RDEP': joblib.load('RDEP_regressor.pkl'),
    'RXO': joblib.load('RXO_regressor.pkl'),
    'RSHA': joblib.load('RSHA_regressor.pkl')
}
def impute_column(df, column_name, model):
    missing_mask = df[column_name].isna()
    if missing_mask.sum() > 0:
        missing_data = df[missing_mask]
        if missing_data.shape[0] > 0:
            features = missing_data.drop(columns=[column_name])
            features = features.fillna(features.median())
            predicted_values = model.predict(features)
            df.loc[missing_mask, column_name] = predicted_values
        else:
            print(f"No missing data found for column '{column_name}'.")
    else:
        print(f"No missing values to impute for column '{column_name}'.")
for column in models.keys():
    if column in df_filtered_model_imputed.columns:
        impute_column(df_filtered_model_imputed, column, models[column])
        
model = joblib.load('lithology_classifier.pkl')

missing_mask = df_filtered_model_imputed['FORCE_2020_LITHOFACIES_LITHOLOGY'].isna()
df_missing = df_filtered_model_imputed[missing_mask]

X_missing = df_missing.drop(columns=['FORCE_2020_LITHOFACIES_LITHOLOGY'])

if not X_missing.empty:
    predicted_values = model.predict(X_missing)

    df_filtered_model_imputed.loc[missing_mask, 'FORCE_2020_LITHOFACIES_LITHOLOGY'] = predicted_values

    print("Missing values predicted")
else:
    print("No missing values found to predict.")

df_result = pd.concat([other_columns, df_filtered_model_imputed], axis=1)

lithology_mapping = {
    30000: 'Sandstone',
    65030: 'Sandstone/Shale',
    65000: 'Shale',
    80000: 'Marl',
    74000: 'Dolomite',
    70000: 'Limestone',
    70032: 'Chalk',
    88000: 'Halite',
    86000: 'Anhydrite',
    99000: 'Tuff',
    90000: 'Coal',
    93000: 'Basement'
}
df_result['FORCE_2020_LITHOFACIES_LITHOLOGY'] = df_result['FORCE_2020_LITHOFACIES_LITHOLOGY'].replace(lithology_mapping)
df_result.to_csv(f'{file_name}_Lithology.csv', index=False)

color_dict = {
    'Sandstone': '#FFA500',       # Orange (commonly used for sandstone)
    'Sandstone/Shale': '#A52A2A', # Brownish-red (to represent a mix of sandstone and shale)
    'Shale': '#2E8B57',           # Sea green (a deeper green for shale)
    'Marl': '#FFFF66',            # Light yellow (typical for marl)
    'Dolomite': '#4682B4',        # Steel blue (used for dolomite)
    'Limestone': '#8B0000',       # Dark red (contrasting color for limestone)
    'Chalk': '#FFD700',           # Gold (bright color for chalk)
    'Halite': '#D2691E',          # Chocolate brown (for halite)
    'Anhydrite': '#8B4513',       # Saddle brown (stays the same as it's well-suited)
    'Tuff': '#800080',            # Purple (dark and distinctive for tuff)
    'Coal': '#2F4F4F',            # Dark slate gray (a slightly softer black for coal)
    'Basement': '#4B0082'         # Indigo (dark and deep for basement rocks)
}
df_sorted = df_result.sort_values('DEPT')
plt.figure(figsize=(8, 12))
plt.barh(df_sorted['DEPT'], [1]*len(df_sorted), color=df_sorted['FORCE_2020_LITHOFACIES_LITHOLOGY'].map(color_dict))
plt.xlabel('Lithology')
plt.ylabel('Depth')
plt.gca().invert_yaxis()
legend_patches = [mpatches.Patch(color=color, label=lithology) for lithology, color in color_dict.items()]
plt.legend(handles=legend_patches, title="Lithology", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Lithology Distribution Across Depth')
plt.show()