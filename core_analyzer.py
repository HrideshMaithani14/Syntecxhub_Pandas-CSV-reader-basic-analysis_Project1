"""
Syntecxhub Data Science Internship - Project 2
Project Title: Pandas CSV Reader & Basic Analysis
Author: Hridesh Maithani

Description:
A modular, production-ready Data Analysis Utility using Pandas that:
1. Ingests CSV and Excel files into DataFrames and inspects head, tail, dtypes, nulls, and shape.
2. Computes core summary statistics (mean, median, min, max, count, std, quartiles).
3. Filters rows dynamically, selects specified columns, and slices DataFrame subsets.
4. Exports filtered and sliced subsets into CSV or Excel (.xlsx) formats.
Includes both a reusable class API and an interactive Command Line Interface (CLI).
"""

import os
import sys
import argparse
from typing import Union, List, Optional, Dict, Any
import pandas as pd


class PandasDataAnalyzer:
    """
    A comprehensive data exploration and manipulation utility powered by Pandas.
    """

    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize the analyzer with an optional file path.
        """
        self.filepath: Optional[str] = filepath
        self.df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None

        if filepath:
            self.load_data(filepath)

    # =========================================================================
    # REQUIREMENT 1: Read CSV/Excel into DataFrame & Inspect Head/Tail/Types
    # =========================================================================
    def load_data(
        self,
        filepath: str,
        sheet_name: Optional[Union[str, int]] = 0,
        delimiter: Optional[str] = None,
        encoding: str = "utf-8"
    ) -> pd.DataFrame:
        """
        Read a CSV or Excel file into a pandas DataFrame.
        Auto-detects format based on file extension.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: '{filepath}'")

        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext in [".csv", ".txt"]:
                sep = delimiter if delimiter else ","
                self.df = pd.read_csv(filepath, sep=sep, encoding=encoding)
            elif ext == ".tsv":
                sep = delimiter if delimiter else "\t"
                self.df = pd.read_csv(filepath, sep=sep, encoding=encoding)
            elif ext in [".xlsx", ".xls", ".xlsm"]:
                self.df = pd.read_excel(filepath, sheet_name=sheet_name)
            else:
                # Attempt CSV fallback
                self.df = pd.read_csv(filepath, encoding=encoding)

            self.filepath = filepath
            self.filtered_df = self.df.copy()
            print(f"[SUCCESS] Loaded '{filepath}' successfully.")
            print(f"          Total Rows: {self.df.shape[0]} | Total Columns: {self.df.shape[1]}")
            return self.df

        except Exception as e:
            raise RuntimeError(f"Error reading file '{filepath}': {e}")

    def inspect_data(self, n: int = 5) -> Dict[str, Any]:
        """
        Inspect dataset characteristics: head, tail, dtypes, nulls, memory, and shape.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        head_preview = self.df.head(n)
        tail_preview = self.df.tail(n)

        type_info = pd.DataFrame({
            "Data_Type": self.df.dtypes.astype(str),
            "Non_Null_Count": self.df.notnull().sum(),
            "Null_Count": self.df.isnull().sum(),
            "Null_Percentage": (self.df.isnull().mean() * 100).round(2),
            "Unique_Values": self.df.nunique()
        })

        summary_info = {
            "shape": self.df.shape,
            "total_rows": self.df.shape[0],
            "total_columns": self.df.shape[1],
            "columns": list(self.df.columns),
            "memory_usage_kb": round(self.df.memory_usage(deep=True).sum() / 1024, 2),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "dtypes_table": type_info,
            "head": head_preview,
            "tail": tail_preview
        }

        return summary_info

    def print_inspection(self, n: int = 5):
        """Pretty-print the data inspection report."""
        info = self.inspect_data(n)
        print("\n" + "=" * 80)
        print(f"📊 DATA INSPECTION REPORT: {os.path.basename(self.filepath or 'InMemory')}")
        print("=" * 80)
        print(f"• Dataset Dimensions: {info['total_rows']} rows x {info['total_columns']} columns")
        print(f"• Total Memory Usage: {info['memory_usage_kb']} KB")
        print(f"• Duplicate Rows:     {info['duplicate_rows']}")
        print("\n--- COLUMN ATTRIBUTES & DATA TYPES ---")
        print(info["dtypes_table"].to_string())

        print(f"\n--- TOP {n} ROWS (HEAD) ---")
        print(info["head"].to_string())

        print(f"\n--- BOTTOM {n} ROWS (TAIL) ---")
        print(info["tail"].to_string())
        print("=" * 80 + "\n")

    # =========================================================================
    # REQUIREMENT 2: Compute Summary Stats (mean, median, min, max, count)
    # =========================================================================
    def compute_summary_stats(
        self,
        columns: Optional[List[str]] = None,
        use_filtered: bool = False
    ) -> pd.DataFrame:
        """
        Compute comprehensive summary statistics focusing on:
        - Mean
        - Median
        - Min
        - Max
        - Count
        along with Std Dev, 25%, 75% quartiles, and IQR.
        """
        target_df = self.filtered_df if use_filtered and self.filtered_df is not None else self.df
        if target_df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Select numeric columns
        if columns:
            valid_cols = [c for c in columns if c in target_df.columns and pd.api.types.is_numeric_dtype(target_df[c])]
        else:
            valid_cols = target_df.select_dtypes(include=["number"]).columns.tolist()

        if not valid_cols:
            print("[WARN] No numerical columns found for summary statistics.")
            return pd.DataFrame()

        stats_dict = {}
        for col in valid_cols:
            series = target_df[col].dropna()
            cnt = int(series.count())
            mean_val = float(series.mean()) if cnt > 0 else float("nan")
            median_val = float(series.median()) if cnt > 0 else float("nan")
            min_val = float(series.min()) if cnt > 0 else float("nan")
            max_val = float(series.max()) if cnt > 0 else float("nan")
            std_val = float(series.std()) if cnt > 1 else float("nan")
            q25 = float(series.quantile(0.25)) if cnt > 0 else float("nan")
            q75 = float(series.quantile(0.75)) if cnt > 0 else float("nan")
            iqr = q75 - q25 if cnt > 0 else float("nan")

            stats_dict[col] = {
                "Count": cnt,
                "Mean": round(mean_val, 4),
                "Median": round(median_val, 4),
                "Min": round(min_val, 4),
                "Max": round(max_val, 4),
                "Std_Dev": round(std_val, 4),
                "Q25 (25%)": round(q25, 4),
                "Q75 (75%)": round(q75, 4),
                "IQR": round(iqr, 4)
            }

        stats_df = pd.DataFrame(stats_dict).T
        return stats_df

    def compute_categorical_summary(
        self,
        columns: Optional[List[str]] = None,
        use_filtered: bool = False
    ) -> pd.DataFrame:
        """
        Compute summary statistics for non-numeric/categorical columns:
        Count, Unique, Mode (most frequent), Mode Frequency.
        """
        target_df = self.filtered_df if use_filtered and self.filtered_df is not None else self.df
        if target_df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if columns:
            cat_cols = [c for c in columns if c in target_df.columns and not pd.api.types.is_numeric_dtype(target_df[c])]
        else:
            cat_cols = target_df.select_dtypes(exclude=["number"]).columns.tolist()

        if not cat_cols:
            return pd.DataFrame()

        summary = {}
        for c in cat_cols:
            s = target_df[c].dropna()
            mode_val = s.mode().iloc[0] if not s.empty else "N/A"
            mode_freq = int((s == mode_val).sum()) if not s.empty else 0
            summary[c] = {
                "Count": int(s.count()),
                "Unique": int(s.nunique()),
                "Top_Mode": str(mode_val),
                "Mode_Freq": mode_freq,
                "Null_Count": int(target_df[c].isnull().sum())
            }

        return pd.DataFrame(summary).T

    def print_summary_stats(self, columns: Optional[List[str]] = None, use_filtered: bool = False):
        """Pretty-print statistical summary tables."""
        numeric_stats = self.compute_summary_stats(columns=columns, use_filtered=use_filtered)
        cat_stats = self.compute_categorical_summary(columns=columns, use_filtered=use_filtered)

        print("\n" + "=" * 80)
        source_label = "FILTERED SUBSET" if use_filtered else "FULL DATASET"
        print(f"📈 NUMERICAL SUMMARY STATISTICS ({source_label})")
        print("=" * 80)
        if not numeric_stats.empty:
            print(numeric_stats.to_string())
        else:
            print("No numerical columns found.")

        if not cat_stats.empty:
            print("\n" + "-" * 80)
            print(f"📋 CATEGORICAL SUMMARY STATISTICS ({source_label})")
            print("-" * 80)
            print(cat_stats.to_string())
        print("=" * 80 + "\n")

    # =========================================================================
    # REQUIREMENT 3: Filter Rows, Select Columns & Slice Subsets
    # =========================================================================
    def filter_by_condition(self, column: str, operator: str, value: Any) -> pd.DataFrame:
        """
        Filter rows based on a single condition:
        Operators supported: '==', '!=', '>', '>=', '<', '<=', 'contains', 'in', 'not in'.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        col_data = self.df[column]

        if operator == "==":
            mask = col_data == value
        elif operator == "!=":
            mask = col_data != value
        elif operator == ">":
            mask = col_data > float(value)
        elif operator == ">=":
            mask = col_data >= float(value)
        elif operator == "<":
            mask = col_data < float(value)
        elif operator == "<=":
            mask = col_data <= float(value)
        elif operator == "contains":
            mask = col_data.astype(str).str.contains(str(value), case=False, na=False)
        elif operator == "in":
            items = value if isinstance(value, list) else [value]
            mask = col_data.isin(items)
        elif operator == "not in":
            items = value if isinstance(value, list) else [value]
            mask = ~col_data.isin(items)
        else:
            raise ValueError(f"Unsupported operator '{operator}'")

        self.filtered_df = self.df[mask].copy()
        print(f"[FILTER] Applied condition: {column} {operator} {value} -> {len(self.filtered_df)} rows matched.")
        return self.filtered_df

    def filter_by_query(self, query_string: str) -> pd.DataFrame:
        """
        Filter rows using Pandas query expression syntax (e.g., "Salary > 100000 and Remote_Work == True").
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        try:
            self.filtered_df = self.df.query(query_string).copy()
            print(f"[QUERY] Query '{query_string}' matched {len(self.filtered_df)} rows.")
            return self.filtered_df
        except Exception as e:
            raise ValueError(f"Invalid query expression '{query_string}': {e}")

    def select_columns(self, columns: List[str], use_filtered: bool = False) -> pd.DataFrame:
        """
        Project and keep only the specified list of columns.
        """
        target_df = self.filtered_df if use_filtered and self.filtered_df is not None else self.df
        if target_df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        missing = [c for c in columns if c not in target_df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        self.filtered_df = target_df[columns].copy()
        print(f"[SELECT] Selected {len(columns)} columns: {columns}")
        return self.filtered_df

    def slice_subset(
        self,
        row_start: Optional[int] = None,
        row_end: Optional[int] = None,
        col_start: Optional[int] = None,
        col_end: Optional[int] = None,
        use_filtered: bool = False
    ) -> pd.DataFrame:
        """
        Slice a subset using positional iloc indexing: rows [row_start:row_end], columns [col_start:col_end].
        """
        target_df = self.filtered_df if use_filtered and self.filtered_df is not None else self.df
        if target_df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        r_slice = slice(row_start, row_end)
        c_slice = slice(col_start, col_end)

        self.filtered_df = target_df.iloc[r_slice, c_slice].copy()
        print(f"[SLICE] Sliced rows [{row_start}:{row_end}], cols [{col_start}:{col_end}] -> Shape: {self.filtered_df.shape}")
        return self.filtered_df

    def reset_filter(self):
        """Reset filtered DataFrame to the original full dataset."""
        if self.df is not None:
            self.filtered_df = self.df.copy()
            print("[RESET] Filtered DataFrame reset to full dataset.")

    # =========================================================================
    # REQUIREMENT 4: Save Filtered Results to CSV / Excel
    # =========================================================================
    def save_results(
        self,
        output_path: str,
        df_to_save: Optional[pd.DataFrame] = None,
        file_format: Optional[str] = None,
        include_index: bool = False,
        sheet_name: str = "Analysis_Results"
    ) -> str:
        """
        Save the filtered or modified DataFrame to a CSV or Excel file.
        Auto-detects output format from file extension or format argument.
        """
        target_df = df_to_save if df_to_save is not None else self.filtered_df
        if target_df is None:
            target_df = self.df

        if target_df is None:
            raise ValueError("No data available to save.")

        # Determine target directory and ensure it exists
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Detect format
        ext = os.path.splitext(output_path)[1].lower()
        fmt = (file_format or ("csv" if ext in [".csv", ""] else "excel" if ext in [".xlsx", ".xls"] else "csv")).lower()

        if fmt in ["csv", ".csv"]:
            if not output_path.lower().endswith(".csv"):
                output_path += ".csv"
            target_df.to_csv(output_path, index=include_index, encoding="utf-8")
        elif fmt in ["excel", "xlsx", ".xlsx", ".xls"]:
            if not (output_path.lower().endswith(".xlsx") or output_path.lower().endswith(".xls")):
                output_path += ".xlsx"
            target_df.to_excel(output_path, index=include_index, sheet_name=sheet_name)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Choose 'csv' or 'excel'.")

        file_size_kb = round(os.path.getsize(output_path) / 1024, 2)
        print(f"[EXPORT] Successfully saved {len(target_df)} rows and {len(target_df.columns)} columns.")
        print(f"         Output File: {os.path.abspath(output_path)} ({file_size_kb} KB)")
        return output_path


# =============================================================================
# INTERACTIVE CLI INTERFACE
# =============================================================================
def interactive_cli():
    """Run an interactive console menu for the Pandas CSV/Excel Analyzer."""
    print("=" * 80)
    print("   SYNTECXHUB INTERNSHIP PROGRAM - PROJECT 2: PANDAS CSV & EXCEL ANALYZER   ")
    print("=" * 80)

    analyzer = PandasDataAnalyzer()

    # Default sample path check
    default_csv = os.path.join(os.path.dirname(__file__), "sample_data", "sample_employees.csv")

    while True:
        if analyzer.df is None:
            print("\n[STEP 1: LOAD DATASET]")
            print(f"Enter file path (.csv, .xlsx, .xls) [Press Enter for default: '{default_csv}']: ")
            path_input = input("Path > ").strip().strip('"\'')
            target_path = path_input if path_input else default_csv

            try:
                analyzer.load_data(target_path)
            except Exception as e:
                print(f"[ERROR] Failed to load dataset: {e}")
                continue

        print("\n" + "-" * 50)
        print("MAIN MENU - Select an Operation:")
        print("1. Inspect Data (Head, Tail, Dtypes, Shape, Nulls)")
        print("2. Compute Summary Statistics (Mean, Median, Min, Max, Count)")
        print("3. Filter Rows by Condition")
        print("4. Filter Rows by Query Expression")
        print("5. Select Specific Columns")
        print("6. Slice Subsets (Positional Row/Column Range)")
        print("7. Preview Current Filtered Dataset")
        print("8. Reset Filter to Original Dataset")
        print("9. Save / Export Filtered Results (CSV or Excel)")
        print("10. Load a Different Dataset")
        print("0. Exit")
        print("-" * 50)

        choice = input("Enter choice (0-10): ").strip()

        if choice == "1":
            n_rows = input("Number of rows to view (default 5): ").strip()
            n = int(n_rows) if n_rows.isdigit() else 5
            analyzer.print_inspection(n=n)

        elif choice == "2":
            use_filt = input("Use filtered subset? (y/N): ").strip().lower() == "y"
            analyzer.print_summary_stats(use_filtered=use_filt)

        elif choice == "3":
            print(f"\nAvailable Columns: {list(analyzer.df.columns)}")
            col = input("Column name: ").strip()
            print("Supported Operators: ==, !=, >, >=, <, <=, contains, in")
            op = input("Operator: ").strip()
            val = input("Value: ").strip()
            try:
                analyzer.filter_by_condition(col, op, val)
                print(f"Current Filtered Rows: {len(analyzer.filtered_df)}")
            except Exception as e:
                print(f"[ERROR] {e}")

        elif choice == "4":
            print("Example query: Salary > 90000 and Remote_Work == True")
            q = input("Query: ").strip()
            try:
                analyzer.filter_by_query(q)
                print(f"Current Filtered Rows: {len(analyzer.filtered_df)}")
            except Exception as e:
                print(f"[ERROR] {e}")

        elif choice == "5":
            print(f"\nAvailable Columns: {list(analyzer.df.columns)}")
            cols_input = input("Enter column names separated by commas: ").strip()
            cols = [c.strip() for c in cols_input.split(",") if c.strip()]
            use_filt = input("Apply to existing filtered data? (y/N): ").strip().lower() == "y"
            try:
                analyzer.select_columns(cols, use_filtered=use_filt)
                print(f"Selected columns preview:\n{analyzer.filtered_df.head(5)}")
            except Exception as e:
                print(f"[ERROR] {e}")

        elif choice == "6":
            print("Enter slice ranges (leave blank for default):")
            r_start = input("Row Start Index: ").strip()
            r_end = input("Row End Index: ").strip()
            c_start = input("Column Start Index: ").strip()
            c_end = input("Column End Index: ").strip()

            rs = int(r_start) if r_start.isdigit() else None
            re = int(r_end) if r_end.isdigit() else None
            cs = int(c_start) if c_start.isdigit() else None
            ce = int(c_end) if c_end.isdigit() else None

            use_filt = input("Slice from current filtered data? (y/N): ").strip().lower() == "y"
            try:
                sliced = analyzer.slice_subset(rs, re, cs, ce, use_filtered=use_filt)
                print(f"Sliced subset preview:\n{sliced.head(5)}")
            except Exception as e:
                print(f"[ERROR] {e}")

        elif choice == "7":
            if analyzer.filtered_df is not None:
                print(f"\nPreview of Current Active Dataset (Shape: {analyzer.filtered_df.shape}):")
                print(analyzer.filtered_df.to_string())
            else:
                print("No data available.")

        elif choice == "8":
            analyzer.reset_filter()

        elif choice == "9":
            out_file = input("Enter output path (e.g. results.csv or results.xlsx): ").strip()
            if not out_file:
                out_file = "filtered_results.csv"
            include_idx = input("Include index column? (y/N): ").strip().lower() == "y"
            try:
                saved = analyzer.save_results(output_path=out_file, include_index=include_idx)
                print(f"[DONE] File exported to: {saved}")
            except Exception as e:
                print(f"[ERROR] Failed to save: {e}")

        elif choice == "10":
            new_path = input("Enter new file path (.csv, .xlsx): ").strip()
            try:
                analyzer.load_data(new_path)
            except Exception as e:
                print(f"[ERROR] {e}")

        elif choice == "0":
            print("\nExiting. Thank you for using Syntecxhub Project 2 Analyzer!")
            break
        else:
            print("Invalid selection. Please choose an option between 0 and 10.")


def main():
    parser = argparse.ArgumentParser(description="Syntecxhub Project 2: Pandas CSV Reader & Basic Analysis")
    parser.add_argument("--file", "-f", type=str, help="Path to input CSV or Excel file")
    parser.add_argument("--inspect", action="store_true", help="Print inspection report (head, tail, types)")
    parser.add_argument("--stats", action="store_true", help="Compute summary statistics (mean, median, min, max, count)")
    parser.add_argument("--query", "-q", type=str, help="Filter query string (e.g., 'Salary > 100000')")
    parser.add_argument("--cols", type=str, help="Comma-separated list of columns to select")
    parser.add_argument("--export", "-e", type=str, help="Export filtered data to CSV or Excel file")

    args = parser.parse_args()

    # If no CLI arguments provided, launch interactive menu
    if not args.file and not args.inspect and not args.stats and not args.query and not args.export:
        interactive_cli()
        return

    analyzer = PandasDataAnalyzer(args.file)

    if args.inspect:
        analyzer.print_inspection()

    if args.query:
        analyzer.filter_by_query(args.query)

    if args.cols:
        col_list = [c.strip() for c in args.cols.split(",")]
        analyzer.select_columns(col_list, use_filtered=True)

    if args.stats:
        analyzer.print_summary_stats(use_filtered=bool(args.query or args.cols))

    if args.export:
        analyzer.save_results(args.export)


if __name__ == "__main__":
    main()
