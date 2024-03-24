import re
from Generator.MusicSynthGen import load_data_from_krn
from utils import check_and_retrieveVocabulary

def erase_numbers_in_tokens_with_equal(tokens):
    return [re.sub(r'(?<=\=)\d+', '', token) for token in tokens]

ytrain = [ ['<bos>'] + sequence + ['<eos>'] for sequence in load_data_from_krn("Data/GrandStaff/partitions_grandstaff/types/train.txt")] 
yval = [ ['<bos>'] + sequence + ['<eos>'] for sequence in load_data_from_krn("Data/GrandStaff/partitions_grandstaff/types/val.txt")]
ytest = [ ['<bos>'] + sequence + ['<eos>'] for sequence in load_data_from_krn("Data/GrandStaff/partitions_grandstaff/types/test.txt")]

fpytrain = [ ['<bos>'] + sequence + ['<eos>'] for sequence in load_data_from_krn("Data/PolishScores/partitions_polishscores/excerpts/fold_0/train.txt", krn_type="ekrn", base_folder="PolishScores")]
fpyval = [ ['<bos>'] + sequence + ['<eos>'] for sequence in load_data_from_krn("Data/PolishScores/partitions_polishscores/excerpts/fold_0/val.txt", krn_type="ekrn", base_folder="PolishScores")]
fpytest = [ ['<bos>'] + sequence + ['<eos>'] for sequence in load_data_from_krn("Data/PolishScores/partitions_polishscores/excerpts/fold_0/test.txt", krn_type="ekrn", base_folder="PolishScores")]

gw2i, gi2w = check_and_retrieveVocabulary([ytrain, yval, ytest], "vocab/", f"GrandStaff", save=False)
fpw2i, fpi2w = check_and_retrieveVocabulary([ytrain, yval, ytest, fpytrain, fpyval, fpytest], "vocab/", f"PolishGlobal", save=True)

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