import os
import csv

CSV_PATH = 'winequality-red.csv'
SQL_PATH = 'winequality-red.sql'

def infer_sql_type(value, colname):
    try:
        float(value)
        if '.' in value or 'pH' in colname or 'density' in colname:
            return 'FLOAT'
        else:
            return 'INTEGER'
    except ValueError:
        return 'VARCHAR(255)'

def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV file {CSV_PATH} not found.")
        return

    with open(CSV_PATH, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        # Replace spaces with underscores in headers for SQL
        sql_headers = [h.replace(' ', '_') for h in headers]
        first_row = next(csv_reader)
        sql_types = []
        for i, value in enumerate(first_row):
            colname = headers[i]
            sql_types.append(infer_sql_type(value, colname))

    with open(SQL_PATH, 'w', encoding='utf-8') as sql_file:
        sql_file.write("CREATE TABLE winequality_red (\n")
        column_defs = []
        for i, header in enumerate(sql_headers):
            column_defs.append(f"    {header} {sql_types[i]}")
        sql_file.write(",\n".join(column_defs))
        sql_file.write("\n);\n")

        with open(CSV_PATH, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # skip header
            row_count = 0
            for row in csv_reader:
                formatted_values = []
                for i, value in enumerate(row):
                    if sql_types[i] == 'FLOAT' or sql_types[i] == 'INTEGER':
                        formatted_values.append(value if value else 'NULL')
                    else:
                        escaped_value = value.replace("'", "''")
                        formatted_values.append(f"'{escaped_value}'")
                sql_file.write(f"INSERT INTO winequality_red ({', '.join(sql_headers)}) VALUES ({', '.join(formatted_values)});\n")
                row_count += 1
    print(f"SQL file created successfully at {SQL_PATH} with {row_count} INSERT statements.")

if __name__ == '__main__':
    main() 