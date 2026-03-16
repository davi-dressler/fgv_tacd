# Duplicate data

If you want to identify and remove duplicate rows in a DataFrame, there are two methods that will help: `duplicated` and `drop_duplicates`. 
Each takes as an argument the columns to use to identify duplicated rows.

-`duplicated` returns a boolean vector whose length is the number of rows, and which indicates whether a row is duplicated.
-`drop_duplicates` removes duplicate rows.

By default, the first observed row of a duplicate set is considered unique, but each method has a keep parameter to specify targets to be kept.

- keep='first' (default): mark / drop duplicates except for the first occurrence.
- keep='last': mark / drop duplicates except for the last occurrence.
- keep=False: mark / drop all duplicates.

# Working with missing data

## Dropping missing data

## Filling missing data