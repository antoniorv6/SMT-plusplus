import os
import seaborn as sns
import matplotlib.pyplot as plt

def count_krn_files(root_folder):
    krn_count = 0
    
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".krn"):
                krn_count += 1
    
    return krn_count

# Replace 'your_folder_path' with the path of the folder you want to start from
authors = os.listdir("grandstaff_dataset")
samples = []

for author in authors:
    root_folder = f'grandstaff_dataset/{author}'

    if os.path.exists(root_folder):
        krn_count = count_krn_files(root_folder)
        samples.append(krn_count)
    else:
        print(f"The folder '{root_folder}' does not exist.")

sns.barplot(x=authors, y=samples)

plt.title('GrandStaff dataset distribution by author')
plt.xlabel('Categories')
plt.ylabel('Values')

plt.savefig('authors-samples.png')