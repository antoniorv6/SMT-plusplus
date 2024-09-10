import re
from rich import progress
from utils.vocab_utils import check_and_retrieveVocabulary

def erase_numbers_in_tokens_with_equal(tokens):
    return [re.sub(r'(?<=\=)\d+', '', token) for token in tokens]

#Define a function that erases whitespace elements in an input list
def erase_whitespace_elements(tokens):
    return [token for token in tokens if token != ""]

def clean_kern(krn, avoid_tokens=['*staff2', '*staff1','*Xped', '*ped', '*Xtuplet', '*tuplet', '*cue', '*Xcue', '*rscale:1/2', '*rscale:1', '*kcancel', '*below']):
    krn = krn.split('\n')
    newkrn = []
    # Remove the lines that contain the avoid tokens
    for idx, line in enumerate(krn):
        if not any([token in line.split('\t') for token in avoid_tokens]):
            #If all the tokens of the line are not '*'
            if not all([token == '*' for token in line.split('\t')]):
                newkrn.append(line.replace("\n", ""))
                
    return "\n".join(newkrn)

def load_kern_file(path: str) -> str:
    with open(path, 'r') as file:
        krn = file.read()
        krn = clean_kern(krn.replace('*tremolo', '*').replace('*Xtremolo', '*'))
        krn = krn.replace(" ", " <s> ")
        krn = krn.replace("\t", " <t> ")
        krn = krn.replace("\n", " <b> ")
        krn = krn.replace("·/", "")
        krn = krn.replace("·\\", "")
        krn = krn.replace('@', ' ')
        krn = krn.replace('·', ' ')
            
        krn = krn.split(" ")[4:]
        krn = [re.sub(r'(?<=\=)\d+', '', token) for token in krn]
        
        return krn

def load_from_files_list(file_ref: list, base_folder:str) -> list:
    files = []
    for file_path in file_ref:
        with open(file_path, 'r') as file:
            files = [line for line in file.read().split('\n') if line != ""]
            return [load_kern_file(base_folder + file) for file in progress.track(files)]

ytrain = [ ['<bos>'] + erase_whitespace_elements(sequence) + ['<eos>'] for sequence in load_from_files_list(["Data/GrandStaff/partitions_grandstaff/types/train.txt"], base_folder="Data/GrandStaff/")] 
yval = [ ['<bos>'] + erase_whitespace_elements(sequence) + ['<eos>'] for sequence in load_from_files_list(["Data/GrandStaff/partitions_grandstaff/types/val.txt"], base_folder="Data/GrandStaff/")]
ytest = [ ['<bos>'] + erase_whitespace_elements(sequence) + ['<eos>'] for sequence in load_from_files_list(["Data/GrandStaff/partitions_grandstaff/types/test.txt"], base_folder="Data/GrandStaff/")]

#fpytrain = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/PolishScores/partitions_polishscores/excerpts/fold_0/train.txt", krn_type="ekrn", tokenization_mode="bekern", base_folder="PolishScores")]
#fpyval = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/PolishScores/partitions_polishscores/excerpts/fold_0/val.txt", krn_type="ekrn", tokenization_mode="bekern", base_folder="PolishScores")]
#fpytest = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/PolishScores/partitions_polishscores/excerpts/fold_0/test.txt", krn_type="ekrn", tokenization_mode="bekern", base_folder="PolishScores")]

fpytrain = [ ['<bos>'] + erase_whitespace_elements(sequence) + ['<eos>'] for sequence in load_from_files_list(["Data/Polish_Scores/partitions_polish_scores/excerpts/fold_0/train.txt"], base_folder="Data/Polish_Scores/")]
fpyval = [ ['<bos>'] + erase_whitespace_elements(sequence) + ['<eos>'] for sequence in load_from_files_list(["Data/Polish_Scores/partitions_polish_scores/excerpts/fold_0/val.txt"], base_folder="Data/Polish_Scores/")]
fpytest = [ ['<bos>'] + erase_whitespace_elements(sequence) + ['<eos>'] for sequence in load_from_files_list(["Data/Polish_Scores/partitions_polish_scores/excerpts/fold_0/test.txt"], base_folder="Data/Polish_Scores/")]
print(len(fpyval))
print(len(fpytrain))
print(len(fpytest))
#fpytrain = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/FPGrandStaff/partitions_fpgrandstaff/excerpts/fold_0/train.txt", krn_type="ekrn", tokenization_mode="ekern", base_folder="FPGrandStaff")]
#fpyval = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/FPGrandStaff/partitions_fpgrandstaff/excerpts/fold_0/val.txt", krn_type="ekrn", tokenization_mode="ekern", base_folder="FPGrandStaff")]
#fpytest = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/FPGrandStaff/partitions_fpgrandstaff/excerpts/fold_0/test.txt", krn_type="ekrn", tokenization_mode="ekern", base_folder="FPGrandStaff")]

gw2i, gi2w = check_and_retrieveVocabulary([ytrain, yval, ytest], "vocab/", f"GrandStaff_Kern", save=False)
fpw2i, fpi2w = check_and_retrieveVocabulary([ytrain, yval, ytest, fpytrain, fpyval, fpytest], "vocab/", f"psk", save=False)

#Compute the maximum, minimum and average sequence length of the sequences between ytrain, yval and ytest
max_len = max([len(sequence) for sequence in ytrain + yval + ytest])
min_len = min([len(sequence) for sequence in ytrain + yval + ytest])
avg_len = sum([len(sequence) for sequence in ytrain + yval + ytest])/len(ytrain + yval + ytest)
print(f"Maximum sequence length: {max_len}")
print(f"Minimum sequence length: {min_len}")
print(f"Average sequence length: {avg_len}")
print(f"Number of tokens in GrandStaff: {len(gw2i)}")

print()
#Do the same for PolishScores
max_len = max([len(sequence) for sequence in fpytrain + fpyval + fpytest])
min_len = min([len(sequence) for sequence in fpytrain + fpyval + fpytest])
avg_len = sum([len(sequence) for sequence in fpytrain + fpyval + fpytest])/len(fpytrain + fpyval + fpytest)
print(f"Maximum sequence length: {max_len}")
print(f"Minimum sequence length: {min_len}")
print(f"Average sequence length: {avg_len}")
print(f"Number of tokens in PolishScores: {len(fpw2i)}")
#print(fpw2i)


#Calculate the number of tokens that are in fpw2i, but not in gw2i
fpw2i_not_gw2i = set(fpw2i.keys()) - set(gw2i.keys())
print(f"Number of tokens in PolishScores that are not in GrandStaff: {len(fpw2i_not_gw2i)}")

#Save fpw2i_not_gw2i to a file
with open("fpw2i_not_gw2i.txt", "w") as file:
    file.write("\n".join(fpw2i_not_gw2i))

#print(fpw2i_not_gw2i)
#Calculate the number of tokens that are both in fpw2i and gw2i
fpw2i_and_gw2i = set(fpw2i.keys()) & set(gw2i.keys())
print(f"Number of tokens in PolishScores that are in GrandStaff: {len(fpw2i_and_gw2i)}")

#Compute the percentage of tokens in PolishScores that are not in GrandStaff
percentage = len(fpw2i_not_gw2i)/len(fpw2i.keys())
print(f"Percentage of tokens in PolishScores that are not in GrandStaff: {percentage}")
# Compute the number of times tokens that are in PolishScores do not appear in GrandStaff
count = 0
total_count = 0
for seq in fpytest:
    for token in seq:
        if token in fpw2i_not_gw2i:
            count += 1
        total_count += 1


print(f"Percentage of frequency of tokens in PolishScores do not appear in GrandStaff: {count/total_count}")



#print(fpw2i_not_gw2i)
#print(fpw2i)