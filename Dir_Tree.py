import ast
import os


def extract_functions_from_file(filepath):
    with open(filepath, "r") as file:
        tree = ast.parse(file.read(), filename=filepath)
    functions = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    return functions


def print_tree_and_functions(directory, prefix=""):
    entries = os.listdir(directory)
    entries.sort()
    total_entries = len(entries)

    for i, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        is_last = i == total_entries - 1
        connector = "└── " if is_last else "├── "

        print(f"{prefix}{connector}{entry}")

        if os.path.isdir(path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree_and_functions(path, new_prefix)
        elif entry.endswith(".py"):
            functions = extract_functions_from_file(path)
            function_prefix = prefix + ("    " if is_last else "│   ")
            for j, func in enumerate(functions):
                func_connector = "└── " if j == len(functions) - 1 else "├── "
                print(f"{function_prefix}{func_connector}Function: {func}")


if __name__ == "__main__":
    root_directory = r"C:\Users\p097220\OneDrive - Alliance\Documents\MDF_TOOLS_PATCH\MDF_controle_qualit-_visuel"  # Change this to your target directory
    print_tree_and_functions(root_directory)
