def get_highest_and_lowest_rows(df, columns):
    """
    Extracts rows with the highest and lowest values for each specified column.

    Parameters:
    df (pd.DataFrame): The input dataset.
    columns (list): A list of column names to find the highest and lowest rows for.

    Returns:
    pd.DataFrame: A DataFrame containing unique rows with highest and lowest values for each column.
    """
    selected_rows = []
    
    for column in columns:
        max_row = df.loc[df[column].idxmax()]
        min_row = df.loc[df[column].idxmin()]
        selected_rows.append(max_row)
        selected_rows.append(min_row)
    
    return pd.DataFrame(selected_rows).drop_duplicates()
