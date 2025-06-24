import os
import zipfile
from io import BytesIO
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import csv


def create_chart(data, chart_type, title, xlabel, ylabel, filename, plots_dir, **kwargs):
    """
    Create a chart of the specified type from the given data.
    """
    plt.figure(figsize=kwargs.get('figsize', (10, 6)))
    data.plot(kind=chart_type, **kwargs.get('plot_options', {}))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=kwargs.get('rotation', 45))
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    if 'grid' in kwargs:
        plt.grid(**kwargs['grid'])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()


def create_top_genres_by_category_plot(df, category_col, plots_dir, filename):
    """
    Creates a figure with subplots showing the top 10 rated genres for each category.
    """
    genre_ratings = (df.assign(genre=df['genres'].str.split(','))
                     .explode('genre')
                     .groupby([category_col, 'genre'], observed=False)['rating']
                     .mean()
                     .drop('unknown', level='genre', errors='ignore'))

    unique_categories = genre_ratings.index.get_level_values(0).unique()

    n_cols = min(3, len(unique_categories))
    n_rows = (len(unique_categories) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    axes = axes.flatten()

    for i, category in enumerate(unique_categories):
        ax = axes[i]
        top_10_genres = genre_ratings.loc[category].nlargest(10)
        top_10_genres.sort_values().plot(kind='barh', ax=ax, color='lightsteelblue')
        ax.set_title(f'Top 10 Genres for: {category}')
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Genre')
        ax.grid(axis='x', alpha=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Top 10 Rated Genres by {category_col.replace("_", " ").title()}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()


def download_and_extract_movielens():
    """
    Download and extract the MovieLens 100k dataset.
    """
    if os.path.exists(os.path.join("data_kd", "ml-100k")):
        print("Dataset already exists. Skipping download.")
        return True

    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    print(f"Downloading MovieLens 100k dataset from {url}...")

    os.makedirs("data_kd", exist_ok=True)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to download dataset: {e}")
        return False

    print("Extracting dataset...")
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("data")

    print("Dataset downloaded and extracted successfully.")
    return True


def process_movielens_data():
    """
    Process the MovieLens 100k dataset and create a combined dataframe.
    """
    data_dir = os.path.join("data_kd", "ml-100k")

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        return None

    genre_columns = ['unknown', 'Action', 'Adventure', 'Animation',
                     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                     'Thriller', 'War', 'Western']

    ratings_df = pd.read_csv(os.path.join(data_dir, "u.data"), sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies_df = pd.read_csv(os.path.join(data_dir, "u.item"), sep='|', encoding='latin-1',
                              names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_columns)
    users_df = pd.read_csv(os.path.join(data_dir, "u.user"), sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    movies_df['genres'] = movies_df[genre_columns].apply(lambda row: ','.join([genre_columns[i] for i, val in enumerate(row) if val]), axis=1)
    users_df['gender'] = users_df['gender'].map({'M': 'Male', 'F': 'Female'})
    # users_df['age'] = pd.cut(users_df['age'], bins=[0, 15, 22, 40, 60, 100], labels=["Child", "Teen", "Adult", "Middle-Aged", "Senior"])

    movies_df = movies_df[['movie_id', 'title', 'release_date', 'genres']]
    users_df = users_df[['user_id', 'age', 'gender', 'occupation']]

    ratings_df_clean = ratings_df.dropna()
    movies_df_clean = movies_df.dropna()
    users_df_clean = users_df.dropna()

    print(f"\nRows removed from ratings dataset: {len(ratings_df) - len(ratings_df_clean)}")
    print(f"Rows removed from movies dataset: {len(movies_df) - len(movies_df_clean)}")
    print(f"Rows removed from users dataset: {len(users_df) - len(users_df_clean)}")

    ratings_df = ratings_df_clean
    movies_df = movies_df_clean
    users_df = users_df_clean

    combined_df = ratings_df.merge(movies_df, on='movie_id').merge(users_df, on='user_id')
    combined_df.drop(columns=['user_id', 'movie_id', 'timestamp'], inplace=True)

    combined_csv_path = os.path.join("data_kd", "movielens.csv")
    combined_df.to_csv(combined_csv_path, index=False)

    print(f"Combined dataset created with {len(combined_df)} records.")
    return combined_df


def preprocess_and_visualize(df):
    """
    Preprocess the combined dataframe and create visualizations.
    """
    if df is None:
        return

    plots_dir = os.path.join("data_kd", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Plots will be saved to {plots_dir}")

    df['age'] = pd.cut(df['age'], bins=[0, 15, 22, 40, 60, 100], labels=["Child", "Teen", "Adult", "Middle-Aged", "Senior"])

    avg_rating_by_genre = (df.assign(genre=df['genres'].str.split(','))
                           .explode('genre')
                           .groupby('genre')['rating']
                           .mean()
                           .drop('unknown', errors='ignore')
                           .sort_values(ascending=False))

    all_genres = df['genres'].str.split(',').explode()

    visualizations = [
        {'data': df.groupby('gender')['rating'].mean(), 'type': 'bar', 'title': 'Average Rating by Gender', 'xlabel': 'Gender', 'ylabel': 'Average Rating', 'filename': 'avg_rating_by_gender.png', 'kwargs': {'plot_options': {'color': ['lightpink', 'lightblue']}, 'ylim': (0, 5), 'grid': {'axis': 'y', "alpha": 0.5}, 'rotation': 0}},
        {'data': avg_rating_by_genre, 'type': 'bar', 'title': 'Average Rating by Genre', 'xlabel': 'Genre', 'ylabel': 'Average Rating', 'filename': 'avg_rating_by_genre.png', 'kwargs': {'plot_options': {'color': 'purple'}, 'ylim': (0, 5), 'grid': {'axis': 'y', "alpha": 0.75}}},
        {'data': df.groupby('age', observed=False)['rating'].mean().reindex(["Child", "Teen", "Adult", "Middle-Aged", "Senior"]), 'type': 'bar', 'title': 'Average Rating by Age Group', 'xlabel': 'Age Group', 'ylabel': 'Average Rating', 'filename': 'avg_rating_by_age_group.png', 'kwargs': {'plot_options': {'color': 'lightgreen'}, 'ylim': (0, 5), 'grid': {'axis': 'y', "alpha": 0.75}}},
        {'data': df.groupby('occupation')['rating'].mean().sort_values(ascending=False), 'type': 'bar', 'title': 'Average Rating by Occupation', 'xlabel': 'Occupation', 'ylabel': 'Average Rating', 'filename': 'avg_rating_by_occupation.png', 'kwargs': {'plot_options': {'color': 'orange'}, 'ylim': (0, 5), 'grid': {'axis': 'y', "alpha": 0.75}}},
        {'data': all_genres.value_counts().head(10).sort_values(), 'type': 'barh', 'title': 'Top 10 Most Popular Genres', 'xlabel': 'Number of Movies', 'ylabel': 'Genre', 'filename': 'top_10_popular_genres.png', 'kwargs': {'plot_options': {'color': 'lightseagreen'}}},
        {'data': df['title'].value_counts().head(10).sort_values(), 'type': 'barh', 'title': 'Top 10 Most Popular Movies (Most Rated)', 'xlabel': 'Number of Ratings', 'ylabel': 'Movie', 'filename': 'top_10_popular_movies.png', 'kwargs': {'plot_options': {'color': 'lightcoral'}}},
    ]

    for viz in visualizations:
        create_chart(viz['data'], viz['type'], viz['title'], viz['xlabel'], viz['ylabel'], viz['filename'], plots_dir, **viz['kwargs'])

    create_top_genres_by_category_plot(df, 'gender', plots_dir, 'top_genres_by_gender.png')
    create_top_genres_by_category_plot(df, 'age', plots_dir, 'top_genres_by_age_group.png')
    create_top_genres_by_category_plot(df, 'occupation', plots_dir, 'top_genres_by_occupation.png')

    print("Preprocessing and visualization completed.")


def create_sql_file(csv_path):
    """
    Create a SQL file from the movielens.csv file.
    The SQL file will contain CREATE TABLE and INSERT statements.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return False

    sql_path = os.path.splitext(csv_path)[0] + '.sql'
    print(f"Creating SQL file at {sql_path}...")

    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)

        first_row = next(csv_reader)

        sql_types = []
        for i, value in enumerate(first_row):
            column_name = headers[i]
            if column_name in ['rating', 'age']:
                sql_types.append('INTEGER')
            elif column_name in ['release_date']:
                sql_types.append('DATE')
            else:
                sql_types.append('VARCHAR(255)')

    with open(sql_path, 'w', encoding='utf-8') as sql_file:
        sql_file.write("CREATE TABLE movielens (\n")

        column_defs = []
        for i, header in enumerate(headers):
            column_defs.append(f"    {header} {sql_types[i]}")

        sql_file.write(",\n".join(column_defs))
        sql_file.write("\n);\n")

        with open(csv_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)

            row_count = 0

            for row in csv_reader:
                formatted_values = []
                for i, value in enumerate(row):
                    if sql_types[i] == 'FLOAT' or sql_types[i] == 'INTEGER':
                        formatted_values.append(value if value else 'NULL')
                    else:
                        escaped_value = value.replace("'", "''")
                        formatted_values.append(f"'{escaped_value}'")

                sql_file.write(f"INSERT INTO movielens ({', '.join(headers)}) VALUES ({', '.join(formatted_values)});\n")
                row_count += 1

    print(f"SQL file created successfully with {row_count} INSERT statements.")
    return True


def process_celeba_csv(n_rows=None, selected_columns=None):
    """
    Process the CelebA CSV file to keep only the first n_rows and remove specified columns.
    drop_columns: list of column names or indices to drop.
    """
    csv_path = "celeba.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if n_rows:
            df = df.head(n_rows)
        df = df[df['Male'] == -1]
        if selected_columns:
            df = df[selected_columns]
        df.replace(-1, 0, inplace=True)
        df.to_csv(f"new_{csv_path}", index=False)
        print(f"Processed {csv_path}: kept first {n_rows} rows and kept columns: {selected_columns}.")
    else:
        print(f"{csv_path} does not exist.")


if __name__ == '__main__':
    # if download_and_extract_movielens():
    #     movielens_df = process_movielens_data()
    #     preprocess_and_visualize(movielens_df)
    #
    #     csv_path = os.path.join("data", "movielens.csv")
    #     create_sql_file(csv_path)
    # else:
    #     print("Failed to download and extract the dataset.")
    # 
    # selected_columns_men = ["image_id", "Bald", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Straight_Hair", "Wavy_Hair",
    #                     "Smiling", "Goatee", "Mustache", "No_Beard" "Attractive", "Young"]
    # selected_columns_women = ["image_id", "Bald", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Straight_Hair", "Wavy_Hair",
    #                     "Smiling", "Bangs", "Heavy_Makeup", "Wearing_Earrings", "Wearing_Lipstick", "Attractive", "Young"]
    # process_celeba_csv(selected_columns=selected_columns_women)
    pass
