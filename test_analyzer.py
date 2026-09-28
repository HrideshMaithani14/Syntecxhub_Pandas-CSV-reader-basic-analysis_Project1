"""
Test Suite for Project 2: Pandas CSV Reader & Basic Analysis
Author: Hridesh Maithani
"""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from core_analyzer import PandasDataAnalyzer


class TestPandasDataAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(__file__)
        cls.csv_file = os.path.join(cls.base_dir, "sample_data", "sample_employees.csv")
        cls.xlsx_file = os.path.join(cls.base_dir, "sample_data", "sample_employees.xlsx")
        cls.output_dir = os.path.join(cls.base_dir, "output_tests")
        os.makedirs(cls.output_dir, exist_ok=True)

    def test_01_read_csv(self):
        """Verify reading CSV file into DataFrame and checking dimensions."""
        analyzer = PandasDataAnalyzer(self.csv_file)
        self.assertIsNotNone(analyzer.df)
        self.assertEqual(analyzer.df.shape[0], 15)
        self.assertEqual(analyzer.df.shape[1], 9)
        self.assertIn("Salary", analyzer.df.columns)

    def test_02_read_excel(self):
        """Verify reading Excel file into DataFrame and checking dimensions."""
        analyzer = PandasDataAnalyzer(self.xlsx_file)
        self.assertIsNotNone(analyzer.df)
        self.assertEqual(analyzer.df.shape[0], 15)
        self.assertEqual(analyzer.df.shape[1], 9)

    def test_03_inspect_head_tail_types(self):
        """Verify inspection of head, tail, dtypes, nulls."""
        analyzer = PandasDataAnalyzer(self.csv_file)
        info = analyzer.inspect_data(n=3)

        self.assertEqual(len(info["head"]), 3)
        self.assertEqual(len(info["tail"]), 3)
        self.assertIn("Salary", info["columns"])
        self.assertIn("dtypes_table", info)
        self.assertEqual(info["total_rows"], 15)
        self.assertEqual(info["duplicate_rows"], 0)

    def test_04_compute_summary_stats(self):
        """Verify computation of summary statistics (mean, median, min, max, count, std)."""
        analyzer = PandasDataAnalyzer(self.csv_file)
        stats = analyzer.compute_summary_stats(columns=["Salary", "Years_Experience"])

        self.assertIn("Salary", stats.index)
        self.assertIn("Years_Experience", stats.index)

        # Check required statistics
        for metric in ["Count", "Mean", "Median", "Min", "Max", "Std_Dev"]:
            self.assertIn(metric, stats.columns)

        # Accuracy checks
        raw_salary = analyzer.df["Salary"]
        self.assertEqual(stats.loc["Salary", "Count"], raw_salary.count())
        self.assertAlmostEqual(stats.loc["Salary", "Mean"], round(raw_salary.mean(), 4), places=3)
        self.assertEqual(stats.loc["Salary", "Median"], round(raw_salary.median(), 4))
        self.assertEqual(stats.loc["Salary", "Min"], raw_salary.min())
        self.assertEqual(stats.loc["Salary", "Max"], raw_salary.max())

    def test_05_filter_rows(self):
        """Verify conditional filtering on numeric and categorical columns."""
        analyzer = PandasDataAnalyzer(self.csv_file)

        # Numerical comparison: Salary > 100,000
        filtered = analyzer.filter_by_condition("Salary", ">", 100000)
        self.assertTrue((filtered["Salary"] > 100000).all())
        self.assertEqual(len(filtered), 6)

        # Categorical matching: Department == 'Data Science'
        filtered_ds = analyzer.filter_by_condition("Department", "==", "Data Science")
        self.assertEqual(len(filtered_ds), 3)

        # Query syntax: Salary >= 100000 and Remote_Work == True
        filtered_q = analyzer.filter_by_query("Salary >= 100000 and Remote_Work == True")
        self.assertEqual(len(filtered_q), 2)

    def test_06_select_columns_and_slice_subsets(self):
        """Verify column projection and iloc positional slicing."""
        analyzer = PandasDataAnalyzer(self.csv_file)

        # Select columns
        cols = ["Name", "Department", "Salary"]
        selected = analyzer.select_columns(cols)
        self.assertEqual(list(selected.columns), cols)
        self.assertEqual(selected.shape, (15, 3))

        # Positional slicing
        sliced = analyzer.slice_subset(row_start=0, row_end=5, col_start=0, col_end=2)
        self.assertEqual(sliced.shape, (5, 2))

    def test_07_save_filtered_results_csv_and_excel(self):
        """Verify saving filtered results to CSV and Excel."""
        analyzer = PandasDataAnalyzer(self.csv_file)
        analyzer.filter_by_condition("Salary", ">=", 100000)

        csv_out = os.path.join(self.output_dir, "high_earners.csv")
        xlsx_out = os.path.join(self.output_dir, "high_earners.xlsx")

        analyzer.save_results(csv_out)
        analyzer.save_results(xlsx_out)

        self.assertTrue(os.path.exists(csv_out))
        self.assertTrue(os.path.exists(xlsx_out))

        # Validate saved files
        reloaded_csv = pd.read_csv(csv_out)
        reloaded_xlsx = pd.read_excel(xlsx_out)

        self.assertEqual(len(reloaded_csv), 6)
        self.assertEqual(len(reloaded_xlsx), 6)
        self.assertTrue((reloaded_csv["Salary"] >= 100000).all())


if __name__ == "__main__":
    unittest.main()
