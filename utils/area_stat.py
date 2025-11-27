import pandas as pd

FINAL_PATH= "/result/result.csv"
OBJ_NUMBER = "Index"
AREA_HEADER ='Area mm^2'
COLUMNS = ["Size","Amount","Min. Area mm^2","Max. Area mm^2","Mean","Std","Mean+Std","CV", "Median"]


def calculate_mean(df:pd.Series):
    return round(float(df.mean()),2)


def create_row(df:pd.DataFrame, name:str):
    print(df.columns)
    count = [len(df[OBJ_NUMBER])]
    ser = pd.Series(df[AREA_HEADER])

    stats =[
        define_min(ser),
        define_max(ser),
        calculate_mean(ser),
        calculate_std(ser),
        calculate_mean_and_std(ser),
        calculate_cv(ser),
        calculate_median(ser)
    ]

    return [name, count]+stats


def calculate_mean_and_std(ser:pd.Series):
    mean = calculate_mean(ser)
    sd = calculate_std(ser)
    return f"{mean:.2f}±{sd:.2f}"

def calculate_cv(ser:pd.Series):
    mean = calculate_mean(ser)
    sd = pd.Series(ser[AREA_HEADER]).std()
    return round(float(sd/mean), 2)





def process_all(names,rows):
    final_df = pd.DataFrame(columns=COLUMNS)

    for i in range (len(names)):
        print(f"Processing {names[i]}")
        temporal_df = pd.read_csv(names[i])
        ls = create_row(temporal_df, rows[i])
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

