import pandas as pd



AREA_HEADER ="Area mm²"

def calculate_mean(df:pd.DataFrame):
    s = pd.Series(df[AREA_HEADER])
    return round(float(s.mean()),2)


def concat_datasets(name:str):
    xls = pd.ExcelFile(name)
    data = {sheet_name: pd.read_excel(xls, sheet_name=sheet_name) for sheet_name in xls.sheet_names}
    dfs = [x for x in data.values()]
    df = pd.concat(dfs)
    new_nums = [x for x in range(len(df["Object Number"]))]
    df["Object Number"] = new_nums
    df.drop(["Area (pixels)"], axis=1, inplace=True)
    remove_commas(df)
    return df


def create_row(df:pd.DataFrame, name:str):
    ls =[]
    ls.append(name)
    ls.append(len(df["Object Number"]))
    ls.append(calculate_mean(df))
################################################3
    ls.append(round(float(df.std()["Area mm²"]), 2))
    ls.append(calculate_mean_and_std(df))
    ls.append(calculate_cv(df))
    return ls

def calculate_mean_and_std(df:pd.DataFrame):
    mean = float(calculate_mean(df))
    sd = float(pd.Series(df[AREA_HEADER]).std())
    return f"{mean:.2f}±{sd:.2f}"

def calculate_cv(df:pd.DataFrame):
    mean = calculate_mean(df)
    sd = pd.Series(df[AREA_HEADER]).std()
    return round(float(sd/mean), 2)




def remove_commas(df:pd.DataFrame):
    df[AREA_HEADER]=(
        df[AREA_HEADER]
        .astype(str)
        .str.replace(",", "",regex=False)
        .str.replace("%","",regex=False)
        .astype(float)
    )

#
# def create_plot(df:pd.DataFrame):
#     x =df[AREA_HEADER]
#     y =range(len(df))
#     plt.scatter(y, x, c="pink")
#
#     plt.xlabel("Sample")
#     plt.ylabel("Area (mm2)")
#     plt.show()
#
#


def process_all(names,rows):
    final_df = pd.DataFrame()
    #final_df.iloc[0] = []
    df = pd.DataFrame()
    ls =[]
    for i in range (1,len(names)):
        print(f"Processing {names[i]}")
        df = concat_datasets(names[i])
        ls = create_row(df, rows[i])
        final_df.loc[i,"col"] = ls
    return final_df