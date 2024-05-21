import re
from Generator.MusicSynthGen import load_data_from_krn
from utils import check_and_retrieveVocabulary

def erase_numbers_in_tokens_with_equal(tokens):
    return [re.sub(r'(?<=\=)\d+', '', token) for token in tokens]

#Define a function that erases whitespace elements in an input list
def erase_whitespace_elements(tokens):
    return [token for token in tokens if token != ""]

ytrain = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/GrandStaff/partitions_grandstaff/types/train.txt", krn_type="ekrn", tokenization_mode="bekern")] 
yval = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/GrandStaff/partitions_grandstaff/types/val.txt", krn_type="ekrn", tokenization_mode="bekern")]
ytest = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/GrandStaff/partitions_grandstaff/types/test.txt", krn_type="ekrn", tokenization_mode="bekern")]

#fpytrain = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/PolishScores/partitions_polishscores/excerpts/fold_0/train.txt", krn_type="ekrn", tokenization_mode="bekern", base_folder="PolishScores")]
#fpyval = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/PolishScores/partitions_polishscores/excerpts/fold_0/val.txt", krn_type="ekrn", tokenization_mode="bekern", base_folder="PolishScores")]
#fpytest = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/PolishScores/partitions_polishscores/excerpts/fold_0/test.txt", krn_type="ekrn", tokenization_mode="bekern", base_folder="PolishScores")]

fpytrain = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/FPGrandStaff/partitions_fpgrandstaff/excerpts/fold_0/train.txt", krn_type="ekrn", tokenization_mode="ekern", base_folder="FPGrandStaff")]
fpyval = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/FPGrandStaff/partitions_fpgrandstaff/excerpts/fold_0/val.txt", krn_type="ekrn", tokenization_mode="ekern", base_folder="FPGrandStaff")]
fpytest = [ ['<bos>'] + erase_whitespace_elements("".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")) + ['<eos>'] for sequence in load_data_from_krn("Data/FPGrandStaff/partitions_fpgrandstaff/excerpts/fold_0/test.txt", krn_type="ekrn", tokenization_mode="ekern", base_folder="FPGrandStaff")]

gw2i, gi2w = check_and_retrieveVocabulary([ytrain, yval, ytest], "vocab/", f"GrandStaff_BeKern", save=False)
fpw2i, fpi2w = check_and_retrieveVocabulary([ytrain, yval, ytest, fpytrain, fpyval, fpytest], "vocab/", f"Polish_BeKern", save=False)

#Compute the maximum, minimum and average sequence length of the sequences between ytrain, yval and ytest
max_len = max([len(sequence) for sequence in ytrain + yval + ytest])
min_len = min([len(sequence) for sequence in ytrain + yval + ytest])
avg_len = sum([len(sequence) for sequence in ytrain + yval + ytest])/len(ytrain + yval + ytest)
print(f"Maximum sequence length: {max_len}")
print(f"Minimum sequence length: {min_len}")
print(f"Average sequence length: {avg_len}")
print(f"Number of tokens in GrandStaff: {len(gw2i)}")

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
#Calculate the number of tokens that are both in fpw2i and gw2i
fpw2i_and_gw2i = set(fpw2i.keys()) & set(gw2i.keys())
print(f"Number of tokens in PolishScores that are in GrandStaff: {len(fpw2i_and_gw2i)}")

#print(fpw2i_not_gw2i)