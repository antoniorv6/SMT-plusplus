import os

def extract_paths(file_path, num_paths=5):
    paths_by_author = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            parts = line.strip().split('/')
            if len(parts) >= 2:
                author = parts[1]
                if author not in paths_by_author:
                    paths_by_author[author] = []
                paths_by_author[author].append(line.strip())

    selected_paths = []
    print(paths_by_author)

    for author, author_paths in paths_by_author.items():
        num_paths_per_author = min(num_paths, len(author_paths))
        selected_paths.extend(author_paths[:num_paths_per_author])

    return selected_paths

if __name__ == "__main__":
    for i in range(5):
        file_path = f"partitions_fpgrandstaff/excerpts/fold_{i}/test.txt"  # Replace with your file path
        selected_paths = extract_paths(file_path)

        with open(f"subset_fold_{i}.txt", "w") as sel_files:
            sel_files.write("\n".join(selected_paths))