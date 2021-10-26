from one_run import make_worklist_one_run
from split_input import get_sub_exp_input_list, write_sub_exp_input_list
import pandas as pd


def make_worklist_full_2d(exp_input, delimiter_cell, delimiter_col,  # info about experiment input file
                          coord0, coord1,  # coordinates of cells specifying 2 parameters
                          output_dir,  # where the worklists are stored
                          nsub0, nsub1,  # specification of the size in each experiment
                          nrep, npergroup,  # experiment setup info not in the file, for each run
                          reverse_var,  # reverse the importance of the variables
                          dispense_type, asp_mixing,  # liquid handing parameters
                          nzfill,  # shared deck parameter: how the hamilton software adds leading zeroes
                          assay_plate_prefix, nplate, nperplate, ncol, sort_by_col,  # destination setup
                          plate_df, export_intermediate, # source setup
                          time_df, prefix):
    """
    make worklists from experimentel setup
    :param exp_input: dataframe, experimental setup
    :param delimiter_cell: delimiter to separate options of a variable
    :param delimiter_col: delimiter to separate row and col indices of the coordinate, to use in column name of options
    :param coord0: coordinate of the primary variable
    :param coord1: coordinate of the secondary variable
    :param output_dir: output directory
    :param nsub0: sub-number of options for the primary variable
    :param nsub1: sub-number of options for the secondary variable
    :param nrep: number of replicates
    :param npergroup: number of strips per group
    :param reverse_var: reverse the order of variables when sorting
    :param dispense_type: dispense type
    :param asp_mixing: mixing during aspiration
    :param nzfill: number to fill with leading zeros to
    :param assay_plate_prefix: prefix for assay plate names
    :param nplate: number of plates
    :param nperplate: number of strips per plate
    :param ncol: number of columns
    :param sort_by_col: sort by columns
    :param plate_df: dataframe, plates on the instrument
    :param export_intermediate: export intermediate files
    :param time_df: dataframe, time it takes to run steps
    :param prefix: prefix for output filenames
    :return: none
    """
    sub_exp_input_list = get_sub_exp_input_list(exp_input, coord0, coord1,
                                                nsub0, nsub1, delimiter_cell, delimiter_col)

    input_files = write_sub_exp_input_list(sub_exp_input_list, output_dir, prefix)

    for each in input_files:
        output_prefix = each[:-4] + '_'
        temp = make_worklist_one_run(pd.read_csv(each), delimiter_cell, delimiter_col,  # info about experiment input file
                                     nrep, npergroup,  # experiment setup info not in the file
                                     reverse_var,  # reverse the importance of the variables
                                     dispense_type, asp_mixing,  # liquid handing parameters
                                     nzfill,  # shared deck parameter: how the hamilton software adds leading zeroes
                                     assay_plate_prefix, nplate, nperplate, ncol, sort_by_col,  # destination setup
                                     plate_df, output_prefix, export_intermediate,  # source setup
                                     time_df)
