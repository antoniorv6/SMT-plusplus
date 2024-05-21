from musicdiff import diff, DetailLevel, Visualization

#Visualization.INSERTED_COLOR = "green"
#Visualization.DELETED_COLOR = "red"
#Visualization.CHANGED_COLOR = "yellow"

diff("ground_truth.krn", "ground_truth.krn", out_path1="Marked_piano.pdf", out_path2="Marked2.pdf", visualize_diffs=True, detail=DetailLevel.AllObjectsWithStyle)