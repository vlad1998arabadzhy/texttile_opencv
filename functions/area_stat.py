import pandas as pd

FINAL_PATH= "/result/result.csv"
OBJ_NUMBER = "Index"
AREA_HEADER ='Area mm^2.png'
COLUMNS_AREA = ["Size","Amount","Min. Area mm^2","Max. Area mm^2.png","Mean","Std","Mean+Std","CV", "Median"]

COLUMNS_PERIMETER = ["Size","Amount","Min. Perimeter mm","Max. Perimeter mm","Mean","Std","Mean+Std","CV", "Median"]
TRASH = "Trash"
TOO_BIG = "Too big"
TOO_SMALL = "Too small"
NORMAL= "Normal"


def calculate_mean(df:pd.Series):
    return round(float(df.mean()),2)


def create_row(df:pd.DataFrame, name:str, target_feature:str):
    print(df.columns)
    count = [len(df)]
    ser = pd.Series(df[target_feature])

    stats =[
        define_min(ser),
        define_max(ser),
        calculate_mean(ser),
        calculate_std(ser),
        calculate_mean_and_std(ser),
        calculate_cv(ser),
        calculate_median(ser),
        #TODO NEW Need to update columns
        count_categories_members(TRASH, df),
        count_categories_members(TOO_BIG, df),
        count_categories_members(TOO_SMALL,df),
        count_categories_members(NORMAL,df)
    ]

    return [name, count]+stats


def calculate_mean_and_std(ser:pd.Series):
    mean = calculate_mean(ser)
    sd = calculate_std(ser)
    return f"{mean:.2f}±{sd:.2f}"

def calculate_cv(ser:pd.Series):
    mean = calculate_mean(ser)
    sd = calculate_std(ser)
    return round(float(sd/mean), 2)


def count_categories_members(category_name:str, df:pd.DataFrame):
    ser = df['category']
    return ser.count(category_name)





def process_all(names,rows, columns, target_feature):
    final_df = pd.DataFrame(columns=columns)

    for i in range (len(names)):
        print(f"Processing {names[i]}")
        temporal_df = pd.read_csv(names[i])
        ls = create_row(temporal_df, rows[i], target_feature)
        final_df.loc[len(temporal_df)] = ls
    return final_df


def define_min(ser:pd.Series):
    return ser.min()

def define_max(ser:pd.Series):
    return ser.max()

def calculate_median(ser:pd.Series):
    return ser.median()

def calculate_std(ser:pd.Series):
    return ser.std()

def to_final_df(df:pd.DataFrame, final_path):
    df.to_csv(final_path,index=False)

