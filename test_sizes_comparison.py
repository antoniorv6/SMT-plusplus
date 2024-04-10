from data import GraphicCLDataModule
from config_typings import DataConfig, CLConfig

polish_config = DataConfig(data_path= "Data/PolishScores/partitions_polishscores/excerpts/fold_", 
                           synth_path= "Data/GrandStaff/partitions_grandstaff/types/", 
                           vocab_name= "PolishGlobal_Simple", krn_type= "ekrn", 
                           reduce_ratio= 1., base_folder= "PolishScores", file_format= "jpg", fold=0)

cl_config = CLConfig(num_cl_steps= 3, 
                     max_synth_prob= 0.9, 
                     min_synth_prob= 0.2, 
                     increase_steps= 40000, 
                     finetune_steps= 200000, 
                     curriculum_stage_beginning= 2,
                     teacher_forcing_perc=0.2)

polish_data_module = GraphicCLDataModule(polish_config, cl_config, fold=0)

px_train = polish_data_module.train_dataset.x
px_val = polish_data_module.val_dataset.x
px_test = polish_data_module.val_dataset.x

#Get max, min and average image size (in height and width) and print it saying it is from the PolishScores dataset
max_height = max([image.shape[0] for image in px_train + px_val + px_test])
min_height = min([image.shape[0] for image in px_train + px_val + px_test])
avg_height = sum([image.shape[0] for image in px_train + px_val + px_test])/len(px_train + px_val + px_test)
max_width = max([image.shape[1] for image in px_train + px_val + px_test])
min_width = min([image.shape[1] for image in px_train + px_val + px_test])
avg_width = sum([image.shape[1] for image in px_train + px_val + px_test])/len(px_train + px_val + px_test)
print(f"PolishScores dataset:")
print(f"Maximum image height: {max_height}")
print(f"Minimum image height: {min_height}")
print(f"Average image height: {avg_height}")
print(f"Maximum image width: {max_width}")
print(f"Minimum image width: {min_width}")
print(f"Average image width: {avg_width}")

#grandstaff_config = DataConfig(data_path= "Data/FPGrandStaff/partitions_fpgrandstaff/excerpts/fold_", 
#                           synth_path= "Data/GrandStaff/partitions_grandstaff/types/",
#                           base_folder="FPGrandStaff", 
#                           vocab_name= "PolishGlobal_Simple", krn_type= "bekrn", 
#                           reduce_ratio= 0.5, file_format= "png", fold=0)
#
#fpgs_data_module = GraphicCLDataModule(grandstaff_config, cl_config, fold=0)
#
#px_train = fpgs_data_module.train_dataset.x
#px_val = fpgs_data_module.val_dataset.x
#px_test = fpgs_data_module.val_dataset.x
#
##Get max, min and average image size (in height and width) and print it saying it is from the PolishScores dataset
#max_height = max([image.shape[0] for image in px_train + px_val + px_test])
#min_height = min([image.shape[0] for image in px_train + px_val + px_test])
#avg_height = sum([image.shape[0] for image in px_train + px_val + px_test])/len(px_train + px_val + px_test)
#max_width = max([image.shape[1] for image in px_train + px_val + px_test])
#min_width = min([image.shape[1] for image in px_train + px_val + px_test])
#avg_width = sum([image.shape[1] for image in px_train + px_val + px_test])/len(px_train + px_val + px_test)
#print(f"PolishScores dataset:")
#print(f"Maximum image height: {max_height}")
#print(f"Minimum image height: {min_height}")
#print(f"Average image height: {avg_height}")
#print(f"Maximum image width: {max_width}")
#print(f"Minimum image width: {min_width}")
#print(f"Average image width: {avg_width}")





