# pytest prepends this directory to sys.path so tests can `import tiny` (tests/)
# and `from src.models... import ...` (workspace root) at the same time.