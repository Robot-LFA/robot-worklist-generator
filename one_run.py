import numpy as np
import pandas as pd
import itertools
from rearrange_worklist import reorder_groups


#############
# helper functions
#############


def get_plate_type(plate):
    """
    get plate type, ie remote the plate index
    :param plate: string, input plate
    :return: string, plate without index
    """
    plate_split = plate.split('_')
    plate_type = '_'.join(plate_split[:-1])
    return plate_type


def patch_input(exp_input):
    """
    patch exp_input with extra columns
    :param exp_input: dataframe describing experimental setup
    :return: dataframe with extra columns
    """
    # determine step index and step group index based on timing
    exp_input['step_index'] = np.arange(exp_input.shape[0]) + 1
    exp_input['step_group_index'] = exp_input['step_index']
    for i in range(exp_input.shape[0] - 1):
        if exp_input.loc[i, 'time'] == 0:
            exp_input.loc[i + 1, 'step_group_index'] = exp_input.loc[i, 'step_group_index']

    exp_input['previous_step_index'] = exp_input['step_index'] - 1
    no_time_index = np.setdiff1d(np.where(exp_input['time'] <= 0)[0] + 1, [exp_input.shape[0]])
    exp_input.loc[no_time_index, 'previous_step_index'] = 0

    return exp_input


def get_perm_df(exp_input, nrep, delimiter_cell, delimiter_col, reverse_var):
    """
    get dataframe of permutations
    :param exp_input: dataframe, experimental setup
    :param nrep: number of technical replicates
    :param delimiter_cell: delimiter to separate options of a variable
    :param delimiter_col: delimiter to separate row and col indices of the coordinate, to use in column name of options
    :param reverse_var: reverse the order of variables to sort
    :return: dataframe of permutations
    """

    def count_each_cell(cell_string):
        return len(str(cell_string).replace(' ', '').split(delimiter_cell))

    # identify cells with variations
    exp_input_count = exp_input.applymap(count_each_cell)
    multi_list = np.transpose(np.where(exp_input_count > 1))

    # make lists to permutate
    perm_list = [exp_input.iloc[tuple(each)].replace(' ', '').split(delimiter_cell) for each in multi_list]
    perm_col = [delimiter_col.join(list(each.astype(str))) for each in multi_list]
    # sort values in perm_list
    for each in range(len(perm_list)):
        perm_list[each].sort()

    # add in rep
    perm_list = perm_list[::-1] + [list(range(nrep))]
    perm_col = perm_col[::-1] + ['rep']

    if reverse_var:
        # reverse perm_list and perm_col
        perm_list = perm_list[::-1]
        perm_col = perm_col[::-1]

    # make permutation dataframe and add destination
    perm_df = pd.DataFrame(data=list(itertools.product(*perm_list)),
                           columns=perm_col)
    perm_df['destination'] = np.arange(perm_df.shape[0]) + 1

    return perm_df


def get_worklist_from_perm(exp_input, perm_df, npergroup, delimiter_col):
    """
    get worklist from permutations
    :param exp_input: dataframe, experimental setup
    :param perm_df: dataframe of permutations
    :param npergroup: number of strips per group
    :param delimiter_col: delimiter to separate row and col indices of the coordinate, to use in column name of options
    :return: worklist
    """
    # iterate through permutation dataframe to make worklist
    worklist = pd.DataFrame()

    for iperm in range(perm_df.shape[0]):
        perm = perm_df.iloc[iperm, :]

        temp = exp_input.copy()
        # temp['rep'] = perm['rep']
        temp['destination'] = perm['destination']

        for perm_each_col in np.setdiff1d(perm.index.values, ['rep', 'destination']):
            coord = np.array(perm_each_col.split(delimiter_col)).astype(int)
            temp.iloc[tuple(coord)] = perm[perm_each_col]

        worklist = worklist.append(temp, ignore_index=True, sort=False)

    # determine destination group based on the number of strips to do at once
    all_dst = np.sort(worklist['destination'].unique())
    dst_group = np.floor((all_dst - 1) / npergroup).astype(int) + 1
    dst_df = pd.DataFrame(data=np.transpose([all_dst, dst_group]), columns=['destination', 'destination_group'])
    worklist = worklist.merge(dst_df).reset_index(drop=True)
    worklist = worklist.sort_values(['step_group_index', 'destination_group', 'step_index', 'destination'])

    # determine group numbers
    group_df = worklist.loc[:, ['step_index', 'destination_group']].drop_duplicates().reset_index(drop=True)
    group_df['group'] = np.arange(group_df.shape[0]) + 1
    worklist = worklist.merge(group_df).reset_index(drop=True)

    # determine previous group numbers
    previous_group_df = group_df.rename(columns={'step_index': 'previous_step_index', 'group': 'previous_group'})
    # append data for the case of no previous group
    to_append = previous_group_df.copy()
    to_append['previous_step_index'] = 0
    to_append['previous_group'] = 0
    to_append = to_append.drop_duplicates().reset_index(drop=True)
    previous_group_df = previous_group_df.append(to_append, ignore_index=True, sort=False)
    worklist = worklist.merge(previous_group_df).reset_index(drop=True)

    worklist = worklist.sort_values(['step_group_index', 'destination_group', 'step_index', 'destination'])

    return worklist


def get_worklist_full_factorial(exp_input, nrep, npergroup, delimiter_cell, delimiter_col, reverse_var):
    """
    :param exp_input: dataframe describing experimental setup
    :param nrep: number of technical replicates
    :param npergroup: number of strips per group
    :param delimiter_cell: delimiter to separate options of a variable
    :param delimiter_col: delimiter to separate row and col indices of the coordinate, to use in column name of options
    :param reverse_var: reverse the order of variables to sort
    :return: worklist
    """
    exp_input = patch_input(exp_input)
    perm_df = get_perm_df(exp_input, nrep, delimiter_cell, delimiter_col, reverse_var)

    # remove column in perm_df if it is a dummy
    perm_col = np.setdiff1d(perm_df.columns.values, ['rep', 'destination'])
    col_drop = [[each, int(each.split(delimiter_col)[0])]
                for each in perm_col if exp_input['step'][int(each.split(delimiter_col)[0])] == 'dummy']

    if len(col_drop) > 0:
        perm_df = perm_df.drop(labels=np.transpose(col_drop)[0], axis=1)
        exp_input = exp_input.drop(labels=np.transpose(col_drop)[1].astype(np.int), axis=0).reset_index(drop=True)

    for idrop, each in enumerate(col_drop):
        each_index = each[1]
        newcol = perm_df.columns.values
        for icol, eachcol in enumerate(newcol):
            if eachcol not in ['rep', 'destination']:
                coord = np.array(eachcol.split(delimiter_col)).astype(np.int)
                if each_index < coord[0]:
                    coord[0] = coord[0] - 1
                    newcol[icol] = delimiter_col.join(coord.astype(str))
                    perm_df = pd.DataFrame(data=perm_df, columns=newcol)

    # then get worklist
    worklist = get_worklist_from_perm(exp_input, perm_df, npergroup, delimiter_col)
    return {'worklist': worklist,
            'perm_df': perm_df,
            'exp_input': exp_input}


def cleanup_worklist(worklist, dispense_type, asp_mixing):
    """
    clean up worklist
    :param worklist: input worklist
    :param dispense_type: dispense type
    :param asp_mixing: mixing during aspiration
    :return: cleaned worklist
    """
    worklist = worklist.drop(['step_group_index', 'previous_step_index', 'destination_group'], axis=1)
    worklist = worklist.rename(columns={'group': 'group_number',
                                        'previous_group': 'timer_group_check',
                                        'volume': 'volume_ul',
                                        'time': 'timer_delta'})

    worklist['volume_ul'] = worklist['volume_ul'].astype(float)

    # complete worklist with other columns
    worklist['guid'] = worklist['destination']
    worklist['from_path'] = 'some path'
    worklist['asp_mixing'] = asp_mixing
    worklist['dispense_type'] = dispense_type
    worklist['tip_type'] = get_tip_type(worklist['volume_ul'])

    worklist['dispense_type_temp'] = worklist['dispense_type']
    worklist['dispense_type_temp'].replace(regex=True, inplace=True, to_replace='_', value='')
    worklist['liquid_class'] = 'ivl_tip' + worklist['tip_type'].astype(str) + '_' + \
                               worklist['liquid_class'] + '_' + worklist['dispense_type_temp']
    worklist = worklist.drop('dispense_type_temp', axis=1)

    # update asp_mixing based on liquid_class
    # rules:
    # liquid_class does not contain "mix": set to the default
    # liquid_class contains "mix" but not a number after mix: set to 3
    # liquid_class contains "mix" followed by a number: set to that number, and remove that number from liquid_class

    # worklist = worklist.reset_index(drop=True)
    # n_asp_mixing = worklist.loc[worklist['liquid_class'].str.contains('mix'), 'liquid_class'].\
    #         str.split('mix', expand=True).loc[:, 1].str.split('_', expand=True).loc[:, 0].\
    #         convert_objects(convert_numeric=True).fillna(3).astype(int)

    n_asp_mixing = pd.to_numeric(
        worklist.loc[worklist['liquid_class'].str.contains('mix'), 'liquid_class'].\
            str.split('mix', expand=True).loc[:, 1].str.split('_', expand=True)[0],
        errors='coerce')

    worklist.loc[n_asp_mixing.index.values, 'asp_mixing'] = n_asp_mixing.fillna(3).astype(int)

    ind_change_liquid_class = n_asp_mixing.index.values[~n_asp_mixing.isna()]
    temp_split = worklist.loc[ind_change_liquid_class, 'liquid_class'].str.split('mix', expand=True)
    corrected_class = temp_split[0] + 'mix' + '_' + temp_split[1].str.split('_', n=1, expand=True)[1]
    worklist.loc[ind_change_liquid_class, 'liquid_class'] = corrected_class

    return worklist


def get_tip_type(volume, types=[0, 50, 300, 1000]):
    """
    get tip types for each volume in a volume list
    :param volume: list of volumes
    :param types: size of tips
    :return: tip types
    """
    # convert to numpy arrays just in case
    volume = np.array(volume)
    types = np.array(types)
    # make the comparison, the find the indices
    compatible = volume[:, np.newaxis] <= types[np.newaxis, :]
    ind = (1 - compatible).sum(axis=1)
    tip_type = types[ind]
    return tip_type


def get_assay_area_df(assay_plate_prefix, nplate, nperplate, ncol, nzfill, sort_by_col):
    """
    get dataframe about the assay area
    :param assay_plate_prefix: prefix of assay plate
    :param nplate: number of plates
    :param nperplate: number of strips per plate
    :param ncol: number of columns
    :param nzfill: number to add leading zeros
    :param sort_by_col: sort by column
    :return: dataframe describing the assay area
    """
    plates = [assay_plate_prefix + '_' + str(each + 1).zfill(nzfill) for each in range(nplate)]
    wells = np.arange(nperplate) + 1
    assay_area_df = pd.DataFrame(data=list(itertools.product(plates, wells)),
                                 columns=['plate', 'well'])
    assay_area_df['col'] = ((assay_area_df['well'] - 1) % ncol) + 1

    if sort_by_col:
        sort_list = ['plate', 'col', 'well']
    else:
        sort_list = ['plate', 'well']
    assay_area_df = assay_area_df.sort_values(sort_list)
    assay_area_df['destination'] = np.arange(assay_area_df.shape[0]) + 1

    return assay_area_df


def assign_dst(worklist, assay_plate_prefix, nplate, nperplate, ncol, nzfill, sort_by_col):
    """
    assign destinations
    :param worklist: input worklist
    :param assay_plate_prefix: prefix of assay plate
    :param nplate: number of plates
    :param nperplate: number of stips per plate
    :param ncol: number of columns
    :param nzfill: number to add leading zeros
    :param sort_by_col: sort by column
    :return: worklist with assigned destinations (strip locations)
    """
    assay_area_df = get_assay_area_df(assay_plate_prefix=assay_plate_prefix,
                                      nplate=nplate,
                                      nperplate=nperplate,
                                      ncol=ncol,
                                      nzfill=nzfill,
                                      sort_by_col=sort_by_col)
    assay_area_df = assay_area_df.loc[:, ['plate', 'well', 'destination']].rename(columns={'plate': 'to_plate',
                                                                                           'well': 'to_well'})
    worklist = worklist. \
        reset_index(). \
        drop('index', axis=1). \
        reset_index(). \
        merge(assay_area_df). \
        sort_values('index').drop('index', axis=1)

    return worklist


def assign_src(worklist, plate_df, nzfill):
    """
    assign sources
    :param worklist: input worklist
    :param plate_df: dataframe describing plates on the instrument
    :param nzfill: number to fill with leading zeros to
    :return: worklist with sources
    """
    # first tally up the total volume
    source_df = worklist.groupby('source')['volume_ul'].sum().to_frame().reset_index()
    step_df = worklist.loc[:, ['source', 'step_index', 'step']].drop_duplicates()
    source_df = source_df.merge(step_df).sort_values(['source'])
    source_df = source_df[source_df['volume_ul'] > 0]

    # reagent plate df
    plate_df['volume_usable'] = plate_df['volume_well'] - plate_df['volume_holdover']
    plate_df = plate_df.sort_values('volume_usable')

    # assign plates
    source_df['volume_usable'] = get_tip_type(source_df['volume_ul'], np.append([0], plate_df['volume_usable'].values))
    source_df = source_df.merge(plate_df)

    # assign wells
    # go through each step, plate combo and assign well numbers
    source_df['plate_well'] = 0
    for each_step in source_df['step'].unique():
        for each_plate in source_df['plate'].unique():
            sub_df = source_df[(source_df['step'] == each_step) & (source_df['plate'] == each_plate)]
            if sub_df.shape[0] > 0:
                nrow = sub_df['nrow'].values[0]
                plate_well = np.arange(sub_df.shape[0])
                if '384' in each_plate:
                    # split into different columns
                    ncol_each = int(np.ceil(sub_df.shape[0] / nrow))
                    iwell = np.arange(nrow * ncol_each).reshape((ncol_each, -1, 2)).swapaxes(-1, -2).flatten()
                    plate_well = [plate_well[each * nrow:(each + 1) * nrow]
                                  for each in range(int(np.ceil(plate_well.shape[0] / nrow)))]
                    plate_well = np.concatenate([iwell[each] for each in plate_well])
                plate_well = plate_well + 1
                # shift based on wells that have used
                shift = source_df[source_df['plate'] == each_plate]['plate_well'].max()
                # change the shift to move to another column
                shift = np.ceil(shift / nrow) * nrow
                plate_well = plate_well + shift
                source_df.loc[sub_df.index.values, 'plate_well'] = plate_well
    source_df['plate_well'] = source_df['plate_well'].astype(int)
    source_df['plate_index'] = np.ceil(source_df['plate_well'] / source_df['ncol'] / source_df['nrow']).astype(int)
    source_df['from_plate'] = source_df['plate'] + '_' + source_df['plate_index'].astype(str).str.zfill(nzfill)
    source_df['from_well'] = source_df['plate_well']
    source_df_out = source_df.copy()
    source_df = source_df.loc[:, ['source', 'from_plate', 'from_well']]

    worklist = worklist. \
        reset_index(). \
        drop('index', axis=1). \
        reset_index(). \
        merge(source_df, how='outer'). \
        sort_values('index').drop('index', axis=1)

    # special case for imaging
    worklist.loc[worklist['step'] == 'imaging', 'from_plate'] = worklist.loc[worklist['step'] == 'imaging', 'to_plate']
    worklist.loc[worklist['step'] == 'imaging', 'from_well'] = worklist.loc[worklist['step'] == 'imaging', 'to_well']

    # special case for the reservoir
    for each in worklist['group_number'].unique():
        sub = worklist[worklist['group_number'] == each]
        if 'ivl_1_' in sub['from_plate'].values[0]:
            worklist.loc[sub.index.values, 'from_well'] = np.arange(sub.shape[0]) % 8 + 1

    return {'worklist': worklist,
            'source_df': source_df_out}


#############
# major function
#############

def make_worklist_one_run(exp_input, delimiter_cell, delimiter_col,  # info about experiment input file
                          nrep, npergroup,  # experiment setup info not in the file
                          reverse_var,  # reverse the importance of the variables
                          dispense_type, asp_mixing,  # liquid handing parameters
                          nzfill,  # shared deck parameter: how the hamilton software adds leading zeroes
                          assay_plate_prefix, nplate, nperplate, ncol, sort_by_col,  # destination setup
                          plate_df, output_prefix, export_intermediate,  # source setup
                          time_df):
    """
    make worklist for one run
    :param exp_input: dataframe, experimental setup
    :param delimiter_cell: delimiter to separate options of a variable
    :param delimiter_col: delimiter to separate row and col indices of the coordinate, to use in column name of options
    :param nrep: number of replicates
    :param npergroup: number of strips per group
    :param reverse_var: reverse the order of variables to sort
    :param dispense_type: dispense type
    :param asp_mixing: mixing during aspiration
:param nzfill: number to fill with leading zeros to
    :param assay_plate_prefix: prefix for assay plate names
    :param nplate: number of plates
    :param nperplate: number of strips per plate
    :param ncol: number of columns
    :param sort_by_col: sort by columns
    :param plate_df: dataframe, plates on the instrument
    :param output_prefix: prefix for output filenames
    :param export_intermediate: export intermediate file
    :param time_df: dataframe, time it takes to run steps
    :return: dictionary of worklist and source dataframes
    """
    # protocol definition
    # full factorial worklist
    factorial = get_worklist_full_factorial(exp_input=exp_input,
                                            nrep=nrep,
                                            npergroup=npergroup,
                                            delimiter_cell=delimiter_cell,
                                            delimiter_col=delimiter_col,
                                            reverse_var=reverse_var)
    worklist = factorial['worklist']
    worklist_raw = worklist.copy()

    worklist = reorder_groups(worklist, time_df)

    # clean up worklist
    worklist = cleanup_worklist(worklist=worklist, dispense_type=dispense_type, asp_mixing=asp_mixing)
    worklist['touchoff_dis'] = -1  # hard code here because it is always the case when dispensing on LFAs

    # destination assignment
    worklist = assign_dst(worklist=worklist,
                          assay_plate_prefix=assay_plate_prefix,
                          nplate=nplate,
                          nperplate=nperplate,
                          ncol=ncol,
                          nzfill=nzfill,
                          sort_by_col=sort_by_col)

    # source assignment
    source_out = assign_src(worklist=worklist,
                            plate_df=plate_df,
                            nzfill=nzfill)

    worklist = source_out['worklist']
    source_df = source_out['source_df']

    groupby_df = worklist.groupby(by=['from_plate', 'from_well'])
    groupby_v = groupby_df['volume_ul'].sum().reset_index()
    groupby_item = groupby_df['source'].unique().reset_index()
    source_real = groupby_item.merge(groupby_v)
    source_real['source'] = [each[0] for each in source_real['source']]
    source_real = source_real[source_real['volume_ul'] > 0].reset_index(drop=True)
    source_real['plate'] = [get_plate_type(each) for each in source_real['from_plate']]
    source_real = source_real.merge(plate_df, how='left')
    source_real['volume_user_input'] = source_real['volume_ul'] + source_real['volume_holdover']

    if export_intermediate:
        factorial['exp_input'].to_csv(output_prefix + 'exp_input_patched.csv', index=False)
        factorial['perm_df'].to_csv(output_prefix + 'perm_df.csv', index=False)
        worklist_raw.to_csv(output_prefix + 'worklist_raw.csv', index=False)
        worklist.to_csv(output_prefix + 'worklist.csv', index=False)
        source_df_out = source_df.copy()
        source_df_out['volume_total'] = source_df_out['volume_ul'] + source_df_out['volume_holdover']
        source_df_out = source_df_out.loc[:, ['source', 'volume_total', 'from_plate', 'from_well',
                                              'step', 'step_index', 'volume_ul']]
        source_df_out.to_csv(output_prefix + 'source.csv', index=False)
        source_real.to_csv(output_prefix + 'source_real.csv', index=False)

    return {'worklist': worklist,
            'source_df': source_df,
            'source_real': source_real}
