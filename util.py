import pandas as pd
import numpy as np
from scipy.optimize import nnls


def get_source(worklist, plate_df):
    """
    get sources from worklist
    :param worklist: worklist
    :param plate_df: dataframe, plates on the instrument
    :return:
    """
    # get realistic source
    groupby_df = worklist.groupby(by=['from_plate', 'from_well'])
    groupby_v = groupby_df['volume_ul'].sum().reset_index()
    groupby_item = groupby_df['source'].unique().reset_index()
    source_real = groupby_item.merge(groupby_v)
    source_real['source'] = [each[0] for each in source_real['source']]
    source_real = source_real[source_real['volume_ul'] > 0].reset_index(drop=True)
    source_real['plate'] = source_real['from_plate'].str.rsplit('_', n=1, expand=True).iloc[:, 0].values
    source_real = source_real.merge(plate_df, how='left')
    source_real['volume_user_input'] = source_real['volume_ul'] + source_real['volume_holdover']
    return source_real


def get_sol_df(sol_filename):
    """
    get dataframe of solutions
    :param sol_filename: solution file, Excel
    :return: dataframe of solutions
    """
    sol_file = pd.ExcelFile(sol_filename)
    sol_df = pd.concat([pd.read_excel(sol_file, each) for each in sol_file.sheet_names], sort=False).fillna(0).reindex()
    sol_df.index = sol_file.sheet_names
    return sol_df


def get_dilution_df(target, diluent, sol_df, target_volume=None, tolerance=0.01):
    """
    get dataframe showing how the solution is made from sources
    :param target: target solution, string
    :param diluent: diluent
    :param sol_df: dataframe, descriptions of solutions
    :param target_volume: volume of the target solution
    :param tolerance: tolerance of concentrations
    :return: dataframe describing how to make the solution
    """
    if (target_volume==None).any():
        target_volume = np.ones(len(target))

    list_each_make_df = []

    # it may be cleaner to do one at a time
    for each_target, each_volume in zip(target, target_volume):
        target_df = sol_df.reindex([each_target])
        target_df = target_df.loc[:, target_df.gt(0).all()]
        # check if there are ingredients
        ingredients = target_df.columns.values
        each_make_df = pd.DataFrame(data=[[each_volume]], index=[each_target], columns=[each_target])
        if ingredients.shape[0] > 0 and np.in1d(np.append(ingredients, [diluent]), sol_df.columns).all():
            # find appropriate stock solutions
            include = (sol_df[target_df.columns.values] > 0).any(1)
            exclude = (sol_df.drop(target_df.columns.values, axis=1)==0).all(1)
            stock_df = sol_df.loc[include & exclude].drop(each_target)
            stock_df = stock_df.loc[:, (stock_df > 0).any(0)]

            # now add volume and diluent to solve
            target_df['volume'] = 1
            stock_df.loc[diluent, :] = 0
            stock_df['volume'] = 1

            eq_solution = nnls(stock_df.values.transpose(), target_df.values.flatten())
            verify = np.matmul(stock_df.values.transpose(), eq_solution[0])
            verify = (np.abs((verify - target_df.values) / target_df.values) <= tolerance).all()
            if verify:
                each_make_df = pd.DataFrame(data=eq_solution[0]*each_volume.reshape((1, -1)),
                                            index=[each_target], columns=stock_df.index.values)
        list_each_make_df = list_each_make_df + [each_make_df]

    make_df = pd.concat(list_each_make_df, sort=False).round(2).fillna(0)
    make_df[make_df == -0] = 0
    make_df['target'] = make_df.index.values

    return make_df


def get_tip_type(volume, types=[0, 50, 300, 1000]):
    """
    get tip types
    :param volume: list of volumes
    :param types: tip types
    :return: tip types
    """
    # convert to numpy arrays just in case
    volume = np.array(volume)
    types = np.array(types)
    # make the comparison, the find the indices
    compatible = volume[:, np.newaxis] <= types[np.newaxis, :]
    ind = (1-compatible).sum(axis=1)

    # in case the volume is larger than the allowed volume, use the largest one anyway
    tip_type = np.zeros(volume.shape) + types.max()
    i_update = np.where(ind < types.shape[0])[0]
    tip_type[i_update] = types[ind[i_update]]
    return tip_type


def split_transfer(total, each):
    """
    split the transfer of 1 large volume into smaller ones, the remainder is the last transfer
    :param total: total volume
    :param each: max volume each time
    :return: array of volumes
    """
    ratio = total / each
    n_full = np.floor(ratio).astype(int)
    res = ratio - n_full

    out = np.ones(n_full)
    if res > 0:
        out = np.append(out, res)

    out = out * each
    return out


def shift_plate(plate_list):
    """
    shift the plate indices up to make the deck tidier
    :param plate_list: list of plates
    :return: new list of plates
    """
    plate_list = [each.copy() for each in plate_list]

    for idf in range(1, len(plate_list)):
        current_df = plate_list[idf]
        previous_df = pd.concat(plate_list[:idf], sort=False)

        for each_plate in current_df['plate'].unique():
            sub = current_df[current_df['plate'] == each_plate]
            max_previous = previous_df.loc[previous_df['plate'] == each_plate, 'plate_number'].max()
            min_current = sub['plate_number'].min()
            if np.isnan(max_previous):
                max_previous = 0
            current_df.loc[sub.index.values, 'plate_number'] += max_previous - min_current + 1
        plate_list[idf] = current_df  # not really necessary because current_df was not copied (just in case)

    return plate_list


def shift_plate_worklist(worklist0_input, worklist1_input):
    """
    shift the plate indices considering 2 worklists
    :param worklist0_input: worklist0
    :param worklist1_input: worklist1
    :return: new worklists
    """
    worklist0 = worklist0_input.copy()
    worklist1 = worklist1_input.copy()

    to0, n_to_0 = get_plate_type_number(worklist0[['to_plate']])
    from0, n_from_0 = get_plate_type_number(worklist0[['from_plate']])
    to1, n_to_1 = get_plate_type_number(worklist1[['to_plate']])
    from1, n_from_1 = get_plate_type_number(worklist1[['from_plate']])

    to0, from0, to1, from1 = shift_plate([to0, from0, to1, from1])

    worklist0['to_plate'] = to0['plate'] + '_' + to0['plate_number'].astype(str).str.zfill(n_to_0)
    worklist0['from_plate'] = from0['plate'] + '_' + from0['plate_number'].astype(str).str.zfill(n_from_0)
    worklist1['to_plate'] = to1['plate'] + '_' + to1['plate_number'].astype(str).str.zfill(n_to_1)
    worklist1['from_plate'] = from1['plate'] + '_' + from1['plate_number'].astype(str).str.zfill(n_from_1)

    return worklist0, worklist1


def renumber_reservoir(worklist, reservoir_tag='ivl_1_'):
    """
    renumber the wells when a reservoir is used
    :param worklist: input worklist
    :param reservoir_tag: tag for reservoirs
    :return: new worklist
    """
    out = worklist.copy()
    for each in out['group_number'].unique():
        # for from plate
        sub = out[(out['group_number'] == each) & (out['from_plate'].str.contains(reservoir_tag))]
        if sub.shape[0] > 0:
            out.loc[sub.index.values, 'from_well'] = np.arange(sub.shape[0]) % 8 + 1
        # for to plate
        sub = out[(out['group_number'] == each) & (out['to_plate'].str.contains(reservoir_tag))]
        if sub.shape[0] > 0:
            out.loc[sub.index.values, 'to_well'] = np.arange(sub.shape[0]) % 8 + 1
    return out


def assign_plate_well(worklist, plate_df_input, colname, use_holdover=1):
    """
    assign plate and well for transfer steps
    :param worklist: input worklist
    :param plate_df_input: dataframe describing plates
    :param colname: column name
    :param use_holdover: use holdover volume
    :return: dataframe of plates and wells
    """
    # first tally up the total volume
    well_df = worklist.groupby(colname)['volume_ul'].sum().to_frame().reset_index()
    well_df = well_df[well_df['volume_ul'] > 0]

    # reagent plate df
    plate_df = plate_df_input.copy()
    plate_df['volume_usable'] = plate_df['volume_well'] - use_holdover*plate_df['volume_holdover']
    plate_df = plate_df.sort_values('volume_usable')

    # assign plates
    well_df['volume_usable'] = get_tip_type(well_df['volume_ul'], np.append([0], plate_df['volume_usable'].values))
    well_df = well_df.merge(plate_df)
    well_df['nwellperplate'] = well_df['ncol'] * well_df['nrow']

    # now assign location based on the plate
    well_df['plate_number'] = 0
    well_df['well_number'] = 0

    for each_plate in well_df['plate'].unique():
        sub = well_df[well_df['plate'] == each_plate]
        nwellperplate = sub['nwellperplate'].values[0]
        plate_well = np.arange(sub.shape[0])
        well_df.loc[sub.index.values, 'plate_number'] = np.floor(plate_well / nwellperplate) + 1
        well_df.loc[sub.index.values, 'well_number'] = plate_well % nwellperplate + 1

    well_df = well_df[[colname, 'plate', 'plate_number', 'well_number']]
    return well_df


def get_worklist_from_recipe(make_solution_df, tip_size, plate_df, liquid_type_df, n_per_group, nzfill):
    """
    make worklist from recipe
    :param make_solution_df: recipe
    :param tip_size: list of tip sizes, [50, 300, 1000] in most cases
    :param plate_df: dataframe describing plates
    :param liquid_type_df: dataframe, liquid types of solutions
    :param n_per_group: number of transfer steps per group, usually 8 for IVL's Hamilton robots
    :param nzfill: number of digits to fill with leading zeroes to
    :return: new worklist
    """
    # first turn df into transfer list
    worklist = make_solution_df.melt(id_vars='target', var_name='source', value_name='volume_ul')
    worklist = worklist[worklist['volume_ul'] > 0].reset_index(drop=True)
    # remove steps where the target and the source are the same
    worklist = worklist[~(worklist['target'] == worklist['source'])].reset_index(drop=True)

    # split volume if necessary
    i_large_volume = worklist[worklist['volume_ul'] > tip_size.max()].index.values
    for index_to_fix in i_large_volume:
        large_volume = worklist.loc[index_to_fix, 'volume_ul']
        volume_list = split_transfer(large_volume, tip_size.max())
        patch = pd.concat(len(volume_list) * [worklist.loc[i_large_volume]], sort=False)
        patch['volume_ul'] = volume_list
        worklist = pd.concat([worklist.drop(index_to_fix), patch], sort=False)

    # sort
    worklist = worklist.sort_values(['target', 'source', 'volume_ul']).reset_index(drop=True)

    # make the guid column
    worklist['guid'] = worklist['target']

    # below are some hardcoded constants
    asp_mixing = 0
    dispense_type = 'Jet_Empty'
    dx = 0
    dz = 0
    step = 'solution'
    timer_delta = 0
    timer_group_check = 0
    touchoff_dis = 1

    # fill in to make the worklist
    worklist['asp_mixing'] = asp_mixing
    worklist['dispense_type'] = dispense_type
    worklist['dx'] = dx
    worklist['dz'] = dz
    worklist['step'] = step
    worklist['timer_delta'] = timer_delta
    worklist['timer_group_check'] = timer_group_check
    worklist['touchoff_dis'] = touchoff_dis

    # fill in with some calculations
    worklist['from_path'] = worklist['source'] + ' --> ' + worklist['guid']
    worklist['tip_type'] = get_tip_type(worklist['volume_ul'])
    # to get liquid class
    worklist = worklist.merge(liquid_type_df.rename(columns={'solution': 'source'}), how='left').fillna('pbst')
    worklist['liquid_class'] = 'ivl_tip' + worklist['tip_type'].astype(int).astype(str) + '_' + \
                               worklist['liquid_type'] + '_' + worklist['dispense_type'].str.replace('[^a-zA-Z]+', '')

    # get group number
    worklist['group_number'] = 0
    temp = worklist[['guid', 'liquid_class', 'touchoff_dis']].drop_duplicates()
    for irow in range(temp.shape[0]):
        row = temp.iloc[irow]
        sub = worklist[(worklist['guid'] == row['guid']) &
                       (worklist['liquid_class'] == row['liquid_class']) &
                       (worklist['touchoff_dis'] == row['touchoff_dis'])]
        worklist.loc[sub.index.values, 'group_number'] = np.floor(np.arange(sub.shape[0]) / n_per_group) + 1 + \
                                                         worklist['group_number'].max()

    # assign wells
    plate_well_dst = assign_plate_well(worklist, plate_df, colname='guid', use_holdover=0)
    plate_well_src = assign_plate_well(worklist, plate_df, colname='source', use_holdover=1)
    # rearranging plate well, not taking into account overlapping solutions, using different plates in this case
    plate_well_dst, plate_well_src = shift_plate([plate_well_dst, plate_well_src])
    # update wells
    plate_well_src['from_well'] = plate_well_src['well_number']
    plate_well_src['from_plate'] = plate_well_src['plate'] + '_' + plate_well_src['plate_number']. \
        astype(int).astype(str).str.zfill(nzfill)
    worklist = worklist.merge(plate_well_src[['source', 'from_well', 'from_plate']], how='left')

    plate_well_dst['to_well'] = plate_well_dst['well_number']
    plate_well_dst['to_plate'] = plate_well_dst['plate'] + '_' + plate_well_dst['plate_number']. \
        astype(int).astype(str).str.zfill(nzfill)
    worklist = worklist.merge(plate_well_dst[['guid', 'to_well', 'to_plate']], how='left')

    # renumber reservoir
    worklist = renumber_reservoir(worklist)

    return worklist


def get_plate_type_number(plate_df):
    patch = plate_df.iloc[:, 0].str.rsplit('_', n=1, expand=True)
    patch.columns = ['plate', 'plate_number']
    out = plate_df.merge(patch, left_index=True, right_index=True)
    nzfill = out['plate_number'].str.len().max()
    out['plate_number'] = out['plate_number'].astype(int)
    return out, nzfill


def match_from_to_imaging(worklist_input):
    """
    make from and to match for imaging steps
    :param worklist_input: input worklist
    :return: new worklist
    """
    worklist = worklist_input.copy()
    worklist.loc[worklist['step'] == 'imaging', 'from_plate'] = worklist.loc[worklist['step'] == 'imaging', 'to_plate']
    worklist.loc[worklist['step'] == 'imaging', 'from_well'] = worklist.loc[worklist['step'] == 'imaging', 'to_well']
    return worklist


def get_link_sol_run(sol_worklist, run_worklist, tip_size, n_per_group):
    """
    make worklist to link worklists to make solutions and to run assays
    :param sol_worklist: worklist to make solutions
    :param run_worklist: worklist to run assays
    :param tip_size: list of tip sizes, [50, 300, 1000] in most cases
    :param n_per_group: number of steps per group
    :return:
    """
    sol_worklist = sol_worklist.copy()
    run_worklist = run_worklist.copy()

    intersect = np.intersect1d(run_worklist['source'], sol_worklist['target'])

    # go through each in the intersect list
    worklist_list = []
    for each_solution in intersect:
        sol_sub = sol_worklist[sol_worklist['target'] == each_solution]
        run_sub = run_worklist[run_worklist['source'] == each_solution]

        # assumption: there is only 1 well in sol_sub
        from_plate = sol_sub['to_plate'].values[0]
        from_well = sol_sub['to_well'].values[0]

        # also need to tally volume
        worklist = run_sub.groupby(by=['from_plate', 'from_well'])['volume_ul'].sum().reset_index()
        worklist['to_plate'] = worklist['from_plate']
        worklist['to_well'] = worklist['from_well']
        worklist['from_plate'] = from_plate
        worklist['from_well'] = from_well

        # add other information about transferring
        liquid_class_expand = run_sub['liquid_class'].str.split('_', expand=True)
        worklist['liquid_type'] = liquid_class_expand.iloc[0, 2]
        worklist['dispense_type'] = liquid_class_expand.iloc[0, 3]
        worklist['tip_type'] = get_tip_type(worklist['volume_ul'], types=np.append(0, tip_size))
        worklist['liquid_class'] = 'ivl_tip' + worklist['tip_type'].astype(int).astype(str) + '_' + \
                                   worklist['liquid_type'] + '_' + worklist['dispense_type'].str.replace('[^a-zA-Z]+', '')

        worklist_add = {'destination': each_solution,
                        'from_path': each_solution + '_transfer',
                        'guid': each_solution + '_transfer',
                        'source': each_solution,
                        'target': each_solution}
        for each in list(worklist_add.keys()):
            worklist[each] = worklist_add[each]

        worklist_list = worklist_list + [worklist]

    worklist = pd.concat(worklist_list, sort=False).reset_index(drop=True)

    # get group number
    worklist['group_number'] = 0
    temp = worklist[['source', 'liquid_class']].drop_duplicates()
    for irow in range(temp.shape[0]):
        row = temp.iloc[irow]
        sub = worklist[(worklist['source'] == row['source']) &
                       (worklist['liquid_class'] == row['liquid_class'])]
        worklist.loc[sub.index.values, 'group_number'] = np.floor(np.arange(sub.shape[0]) / n_per_group) + 1 + \
                                                         worklist['group_number'].max()

    # finally, add other information
    worklist_add = {'asp_mixing': 0,
                    'dx': 0,
                    'dz': 0,
                    'step': 'solution',
                    'step_index': 0,
                    'timer_delta': 0,
                    'timer_group_check': 0,
                    'touchoff_dis': 1}
    for each in list(worklist_add.keys()):
        worklist[each] = worklist_add[each]

    worklist = renumber_reservoir(worklist)

    return worklist


def worklist_concat(worklist0_input, worklist1_input):
    """
    concatenate worklists
    :param worklist0_input: first worklist
    :param worklist1_input: second worklist
    :return: concatenated worklist
    """
    worklist0 = worklist0_input.copy().reset_index(drop=True)
    worklist1 = worklist1_input.copy().reset_index(drop=True)

    group_shift = worklist0['group_number'].max()
    izero = worklist1[worklist1['timer_group_check'] == 0].index.values
    worklist1[['timer_group_check', 'group_number']] += group_shift
    worklist1.loc[izero, 'timer_group_check'] = 0
    out = pd.concat([worklist0, worklist1], sort=False).reset_index(drop=True)

    return out


def add_plate_well_columns(worklist_input, reservoir_tag='ivl_1'):
    worklist = worklist_input.copy()

    # make temp wells to deal with reservoir
    worklist['_temp_to_well'] = worklist['to_well']
    worklist['_temp_from_well'] = worklist['from_well']

    if reservoir_tag.lower() != 'none':
        worklist.loc[worklist['to_plate'].str.contains('ivl_1'), '_temp_to_well'] = 1
        worklist.loc[worklist['from_plate'].str.contains('ivl_1'), '_temp_from_well'] = 1

    # make plate_well columns
    worklist['to_plate_well'] = worklist['to_plate'] + '|' + worklist['_temp_to_well'].astype(int).astype(str)
    worklist['from_plate_well'] = worklist['from_plate'] + '|' + worklist['_temp_from_well'].astype(int).astype(str)

    return worklist


def consolidate_transfer(worklist_input, keep_tag, reservoir_tag):
    """
    consolidate transfer steps
    :param worklist_input: input worklist
    :param keep_tag: tag of plate_well to keep untouched, usually assay plate tag
    :param reservoir_tag: tag describing the reservoir (1 well in reality, 8 wells in Hamilton software)
    :return: new worklist
    """
    worklist = worklist_input.copy()

    worklist = add_plate_well_columns(worklist, reservoir_tag)

    count_from = worklist.groupby('to_plate_well').apply(lambda df: len(df['from_plate_well'].unique())).reset_index()
    count_from.columns = ['to_plate_well', 'count_from']
    count_from = count_from[(count_from['count_from'] == 1) &
                            (~count_from['to_plate_well'].str.contains(keep_tag))]

    # now go through count_from and update the worklist
    for each in count_from['to_plate_well'].unique():
        each_new = worklist.loc[worklist['to_plate_well'] == each, 'from_plate_well'].values[0]
        # update "from" with each_new
        worklist.loc[worklist['from_plate_well'] == each, 'from_plate_well'] = each_new
        # remove unnecessary transfer rows
        worklist = worklist[~(worklist['to_plate_well'] == each)]

    # update from, to columns
    from_df = worklist['from_plate_well'].str.split('|', expand=True)
    worklist['from_plate'] = from_df.iloc[:, 0]
    worklist['from_well'] = from_df.iloc[:, 1].astype(int)

    to_df = worklist['to_plate_well'].str.split('|', expand=True)
    worklist['to_plate'] = to_df.iloc[:, 0]
    worklist['to_well'] = to_df.iloc[:, 1].astype(int)

    # clean up temp columns
    worklist = worklist.drop(['_temp_to_well', '_temp_from_well', 'from_plate_well', 'to_plate_well'], axis=1)

    # renumber wells for the reservoirs
    worklist = renumber_reservoir(worklist)
    return worklist


def find_plate_info(plate_well, plate_df):
    """
    find plate information
    :param plate_well: plate_well
    :param plate_df: dataframe, plates on the instrument
    :return: plate well dataframe
    """
    plate_well_df = pd.DataFrame.from_dict({'plate_well': plate_well})
    plate_well_df['plate_index'] = plate_well_df['plate_well'].str.split('|', expand=True).iloc[:, 0].values
    plate_well_df['plate'] = plate_well_df['plate_index'].str.rsplit('_', n=1, expand=True).iloc[:, 0].values
    plate_well_df = plate_well_df.merge(plate_df, how='left').drop('plate_index', axis=1)
    return plate_well_df


def update_volume_only(worklist_input, plate_df, reservoir_tag):
    """
    update volume only, to account for holdover volumes
    :param worklist_input: worklist input
    :param plate_df: dataframe, plates on the instrument
    :param reservoir_tag: tag describing the reservoir
    :return: tuple, new worklist and if there has been any changes
    """
    # variable to report of there are changes
    out_change = False

    worklist = worklist_input.copy().reset_index(drop=True)
    worklist0 = worklist.copy()

    worklist = add_plate_well_columns(worklist, reservoir_tag)

    # work on rows with positive transfers only
    worklist = worklist[worklist['volume_ul'] > 0]

    # find hold over volume and max volume

    # get plate info for from_plate_well
    from_plate_info = find_plate_info(worklist['from_plate_well'].values, plate_df)
    from_plate_info = from_plate_info.rename(columns={'plate_well': 'from_plate_well'})
    worklist = worklist.merge(from_plate_info.drop_duplicates(), how='left')

    # go from the bottom, fix from_plate and up
    for irow in worklist.index.values[::-1]:
        row = worklist.loc[irow, :]
        current_plate_well = row['from_plate_well']

        worklist_out = worklist[worklist['from_plate_well'] == current_plate_well]
        worklist_in = worklist[worklist['to_plate_well'] == current_plate_well]

        # find transfer steps before this step only, then scale everything up if necessary
        worklist_out = worklist_out[worklist_out.index <= irow]
        worklist_in = worklist_in[worklist_in.index <= irow]

        if worklist_in.shape[0] > 0:  # if the current_plate_well is not input by the user
            v_out = worklist_out['volume_ul'].sum()
            v_in = worklist_in['volume_ul'].sum()

            # fix holdover issues
            # TODO: come back and make more general. Right now, assume solutions are made before being used
            v_in_scale = (v_out + row['volume_holdover']) / v_in
            if (v_in_scale > 1) and (v_out > 0):
                worklist.loc[worklist_in.index, 'volume_ul'] *= v_in_scale
                out_change = True

    # update worklist0
    col_to_update = np.intersect1d(worklist0.columns.values, worklist.columns.values)
    worklist0.loc[worklist.index, col_to_update] = worklist[col_to_update]

    worklist0 = renumber_reservoir(worklist0)
    return worklist0, out_change


def update_plate_well(worklist_input, plate_df, nzfill, ignore_tag='none', reservoir_tag='none'):
    """
    update plates and wells
    :param worklist_input: input worklist
    :param plate_df: dataframe, plates on the instrument
    :param nzfill: number of digits to fill to, using leading zeroes
    :param ignore_tag: tag to ignore
    :param reservoir_tag: tag describing the reservoir
    :return: new worklist
    """
    worklist = worklist_input.copy()

    # add plate_well_columns
    worklist = add_plate_well_columns(worklist, reservoir_tag=reservoir_tag)

    # again, assume solutions are made before used (nothing going in after something goes out, for each solution)
    # no need to leave space for holdover because it has been taken care of
    new_from_plate_well = assign_plate_well(worklist, plate_df, 'from_plate_well', use_holdover=0)
    new_to_plate_well = assign_plate_well(worklist, plate_df, 'to_plate_well', use_holdover=0)

    if ignore_tag.lower() != 'none':
        new_from_plate_well = new_from_plate_well.loc[~new_from_plate_well['from_plate_well'].str.contains(ignore_tag),
                              :]
        new_to_plate_well = new_to_plate_well.loc[~new_to_plate_well['to_plate_well'].str.contains(ignore_tag), :]

    # update
    new_to_plate_well, new_from_plate_well = shift_plate([new_to_plate_well, new_from_plate_well])

    new_from_plate_well['from_well'] = new_from_plate_well['well_number']
    new_from_plate_well['from_plate'] = new_from_plate_well['plate'] + '_' + new_from_plate_well['plate_number']. \
        astype(int).astype(str).str.zfill(nzfill)
    new_from_plate_well = new_from_plate_well[['from_plate_well', 'from_well', 'from_plate']]
    worklist = worklist.merge(new_from_plate_well, how='left', on='from_plate_well')
    worklist['from_plate'] = worklist['from_plate_y'].fillna(worklist['from_plate_x'])
    worklist['from_well'] = worklist['from_well_y'].fillna(worklist['from_well_x'])
    worklist = worklist.drop(['from_plate_well', 'from_plate_y', 'from_plate_x', 'from_well_y', 'from_well_x'], axis=1)

    new_to_plate_well['to_well'] = new_to_plate_well['well_number']
    new_to_plate_well['to_plate'] = new_to_plate_well['plate'] + '_' + new_to_plate_well['plate_number']. \
        astype(int).astype(str).str.zfill(nzfill)
    new_to_plate_well = new_to_plate_well[['to_plate_well', 'to_well', 'to_plate']]
    worklist = worklist.merge(new_to_plate_well, how='left', on='to_plate_well')
    worklist['to_plate'] = worklist['to_plate_y'].fillna(worklist['to_plate_x'])
    worklist['to_well'] = worklist['to_well_y'].fillna(worklist['to_well_x'])
    worklist = worklist.drop(['to_plate_well', 'to_plate_y', 'to_plate_x', 'to_well_y', 'to_well_x'], axis=1)

    worklist = worklist.drop(['_temp_to_well', '_temp_from_well'], axis=1)

    return worklist


def update_tip_size(worklist_input, tip_size):
    """
    update tip sizes
    :param worklist_input: input worklist
    :param tip_size: tip sizes, [50, 300, 1000] in most cases on the Hamilton
    :return: new worklist
    """
    worklist = worklist_input.copy()
    liquid_class_expand = worklist['liquid_class'].str.split('_', expand=True)
    worklist['liquid_type'] = liquid_class_expand.iloc[0, 2]
    worklist['dispense_type'] = liquid_class_expand.iloc[0, 3]
    worklist['tip_type'] = get_tip_type(worklist['volume_ul'], types=np.append(0, tip_size)).astype(int)
    worklist['liquid_class'] = 'ivl_tip' + worklist['tip_type'].astype(str) + '_' + \
                               worklist['liquid_type'] + '_' + worklist['dispense_type'].str.replace('[^a-zA-Z]+', '')
    worklist['dispense_type'] = worklist['dispense_type'].str.replace('Empty', '_Empty')
    return worklist


def update_holdover_volume_plate_tip(worklist_input, plate_df, nzfill, ignore_tag, reservoir_tag, tip_size, n_iter_max=3):
    """
    update volumes, plates, and tips, to account for hold over volumes
    :param worklist_input: input worklist
    :param plate_df: dataframe, plates on the instrument
    :param nzfill: number of digits to fill to using leading zeros
    :param ignore_tag: tag to ignore
    :param reservoir_tag: tag for the reservoir
    :param tip_size: tip sizes, usually [50, 300, 1000]
    :param n_iter_max: number of maximum iterations
    :return: new worklist
    """
    worklist = worklist_input.copy()
    worklist, vol_change = update_volume_only(worklist, plate_df, reservoir_tag=reservoir_tag)

    n_iter_remaining = n_iter_max
    while vol_change and n_iter_remaining > 0:
        # update plate and well
        worklist = update_plate_well(worklist, plate_df, nzfill=nzfill, ignore_tag=ignore_tag, reservoir_tag=reservoir_tag)
        worklist, vol_change = update_volume_only(worklist, plate_df, reservoir_tag=reservoir_tag)
        n_iter_remaining -= 1

    # update tip size
    worklist = update_tip_size(worklist, tip_size)

    error = 'none'
    if n_iter_remaining == 0 and vol_change == True:
        error = 'cannot update volumes and plates after' + str(1+n_iter_max) + 'iterations'
    if any(worklist['volume_ul'] > tip_size.max()):
        error = 'volume exceeds max tip volume, manually update worklist'

    if error != 'none':
        print('error in update_holdover_volume_plate_tip = ' + error)

    return worklist


def update_dispense_type(worklist_input, ignore_tag, reservoir_tag):
    """
    update dispense type
    :param worklist_input: input worklist
    :param ignore_tag: tag to ignore
    :param reservoir_tag: tag of the reservoir
    :return: new worklist
    """
    worklist = worklist_input.copy().reset_index(drop=True)
    original_columns = worklist.columns.values
    worklist = add_plate_well_columns(worklist, reservoir_tag=reservoir_tag)

    # go top down; change to surface empty if dispense into non-empty wells and to plate is not assay plate
    # assumption: no dispense onto stocks
    for irow in worklist.index.values:
        row = worklist.loc[irow, :]
        if ignore_tag not in row['to_plate']:
            current_plate_well = row['to_plate_well']
            v_in = worklist.loc[(worklist['to_plate_well'] == current_plate_well) & (worklist.index < irow),
                                'volume_ul'].sum()
            if v_in > 0:
                worklist.loc[irow, 'dispense_type'] = worklist.loc[irow, 'dispense_type'].replace('Jet', 'Surface')
                worklist.loc[irow, 'liquid_class'] = worklist.loc[irow, 'liquid_class'].replace('Jet', 'Surface')

    # also ensure that dispense type is the same in each group, with Jet favored over Surface
    for each_group in worklist['group_number'].unique():
        sub = worklist[worklist['group_number'] == each_group]
        if any(sub['dispense_type'].str.contains('Jet')) and any(sub['dispense_type'].str.contains('Surface')):
            worklist.loc[sub.index.values, 'dispense_type'] = worklist.loc[sub.index.values, 'dispense_type']. \
                str.replace('Surface', 'Jet')
            worklist.loc[sub.index.values, 'liquid_class'] = worklist.loc[sub.index.values, 'liquid_class']. \
                str.replace('Surface', 'Jet')

    worklist = worklist[original_columns]
    return worklist


def solution_user_input(worklist_input, plate_df, description_col, reservoir_tag):
    """
    get solution information for the user to put on the instrument
    :param worklist_input: worklist
    :param plate_df: dataframe, plates on the instrument
    :param description_col: description column, such as 'source'
    :param reservoir_tag: tag for the reservoir
    :return: dataframe telling the users which solutions to put where and how much
    """
    worklist = worklist_input.copy()
    worklist = add_plate_well_columns(worklist_input=worklist, reservoir_tag=reservoir_tag)

    vol_from = worklist.groupby('from_plate_well')['volume_ul'].sum().reset_index()
    vol_to = worklist.groupby('to_plate_well')['volume_ul'].sum().reset_index()
    vol_from.columns = ['plate_well', 'vol_from']
    vol_to.columns = ['plate_well', 'vol_to']
    vol = vol_from.merge(vol_to, how='outer').sort_values('plate_well').fillna(0)
    vol['volume_need'] = vol['vol_from'] - vol['vol_to']
    vol = vol[vol['volume_need'] > 0]

    add = vol['plate_well'].str.split('|', expand=True)
    add.columns = ['plate_index', 'well']
    add['plate'] = add['plate_index'].str.rsplit('_', n=1, expand=True).loc[:, 0]

    vol = pd.concat([vol, add], axis=1, sort=False)
    vol = vol.merge(plate_df, how='left')
    vol['user_input'] = vol['volume_need'] + vol['volume_holdover']

    description = worklist.loc[worklist['from_plate_well'].isin(vol['plate_well']),
                               [description_col, 'from_plate_well']].drop_duplicates()
    description.columns = ['solution', 'plate_well']
    vol = vol.merge(description)
    return vol[['solution', 'plate_well', 'user_input']]


def make_solution_worklist(solution_input, diluent, sol_df, liquid_type_df, plate_df, reservoir_tag,
                           ignore_tag, tip_size, n_per_group, nzfill):
    
    """
    make solution worklist
    :param solution_input: solution input, ie. what to make
    :param diluent: diluent, such as water
    :param sol_df: dataframe describing solutions
    :param liquid_type_df: dataframe, liquid types
    :param plate_df: dataframe, plates on the instrument
    :param reservoir_tag: tag for the reservoir
    :param ignore_tag: tag to ignore
    :param tip_size: tip sizes, usually [50, 300, 1000]
    :param n_per_group: number of transfer step per group
    :param nzfill: number of digits to fill to using leading zeroes
    :return: dictionary, including the worklist and dataframes telling the user what to put on the instrument
    """
    make_solution_df = get_dilution_df(target=solution_input['solution'].values,
                                       diluent=diluent,
                                       sol_df=sol_df,
                                       target_volume=solution_input['volume'].values)

    worklist = get_worklist_from_recipe(make_solution_df, tip_size, plate_df, liquid_type_df, n_per_group, nzfill)
    if worklist.shape[0] > 0:
        worklist = update_holdover_volume_plate_tip(worklist, plate_df, nzfill, ignore_tag, reservoir_tag, tip_size)
        worklist = update_dispense_type(worklist, ignore_tag=ignore_tag, reservoir_tag=reservoir_tag)

        user_solution = solution_user_input(worklist, plate_df, 'source', reservoir_tag)
        user_labware = get_labware(worklist, reservoir_tag)
        user_tip = get_tip_count(worklist)
    else:
        user_solution = pd.DataFrame()
        user_labware = pd.DataFrame()
        user_tip = pd.DataFrame()

    return {'worklist': worklist,
            'user_solution': user_solution,
            'user_labware': user_labware,
            'user_tip': user_tip}


def squeeze_plate_index(worklist_input, nzfill):
    """
    squeeze the plate indices down, to consolidate
    :param worklist_input: worklist
    :param nzfill: number of digits to fill to using leading zeroes
    :return: worklist
    """
    worklist = worklist_input.copy().reset_index(drop=True)

    unique_plates = pd.DataFrame.from_dict({'plate': np.unique(worklist[['to_plate', 'from_plate']].values)})
    add = unique_plates['plate'].str.rsplit('_', expand=True, n=1)
    add.columns = ['type', 'index']
    add['index'] = add['index'].astype(int)
    unique_plates = pd.concat([unique_plates, add], axis=1, sort=False)

    for each in unique_plates['type'].unique():
        sub = unique_plates[unique_plates['type'] == each]
        unique_plates.loc[sub.index.values, 'index'] = np.argsort(sub['index'].values) + 1

    unique_plates['plate_new'] = unique_plates['type'] + '_' + unique_plates['index'].astype(str).str.zfill(nzfill)
    unique_plates = unique_plates[~(unique_plates['plate'] == unique_plates['plate_new'])]

    # just loop through, may not be efficient but it works
    for old, new in zip(unique_plates['plate'].values, unique_plates['plate_new'].values):
        worklist.loc[worklist['from_plate'] == old, 'from_plate'] = new
        worklist.loc[worklist['to_plate'] == old, 'to_plate'] = new

    return worklist


def get_labware(worklist, reservoir_tag):
    """
    get worklist to tell the user which labware to use
    :param worklist: worklist
    :param reservoir_tag: tag for reservoirs
    :return: dataframe, labware
    """
    plate_well = add_plate_well_columns(worklist, reservoir_tag)[['from_plate_well', 'to_plate_well']]
    labware = pd.DataFrame(data=np.unique(plate_well.values.flatten()).reshape((-1, 1)), columns=['plate_well'])
    add = labware['plate_well'].str.split('|', expand=True)
    add.columns = ['plate', 'well']
    labware = pd.concat([labware, add], axis=1, sort=False).sort_values(['plate', 'well'])
    return labware


def get_tip_count(worklist):
    """
    get tip count
    :param worklist: worklist
    :return: tip count
    """
    tip_count = []
    for group_number in worklist['group_number'].unique():
        sub = worklist[worklist['group_number']==group_number]
        tip_type = sub['tip_type'].values[0]
        if tip_type > 0:
            tip_group = 'partial' if sub.shape[0] < 8 else 'full'
            each_tip_df = pd.DataFrame(data=[[sub.shape[0]]], columns=['tip_'+str(tip_type)+'_'+tip_group], index=[group_number])
            tip_count = tip_count + [each_tip_df]
    tip_count = pd.concat(tip_count, sort=False).sum(axis=0).astype(int).to_frame(name='count')
    tip_count['tip'] = tip_count.index
    tip_count = tip_count.sort_values('tip')
    return tip_count[['tip', 'count']]


def full_from_run_worklist(run_worklist_input, diluent, sol_df, liquid_type_df, plate_df, reservoir_tag, assay_plate_tag,
                           tip_size, n_per_group, nzfill):
    """
    make full worklist from run worklist
    :param run_worklist_input: run worklist
    :param diluent: diluent
    :param sol_df: dataframe, descriptions of solutions
    :param liquid_type_df: dataframe, liquid types
    :param plate_df: dataframe, plates on the instrument
    :param reservoir_tag: tag for reservoirs
    :param assay_plate_tag: tag for the assay plates
    :param tip_size: tip sizes, usually [50, 300, 1000]
    :param n_per_group: number of steps per group
    :param nzfill: number of digits to fill to using leading zeroes
    :return: dictionary, including worklist and info for the user to put solutions, labware, and tips on
    """
    run_worklist = run_worklist_input.copy()

    source = get_source(run_worklist, plate_df)
    source = source.rename(columns={'volume_user_input': 'volume'})

    # get unique solutions
    source_unique = source.groupby(by='source')['volume'].sum().reset_index().rename(columns={'source': 'solution'})

    input_dict = {'diluent': diluent,
                  'sol_df': sol_df,
                  'liquid_type_df': liquid_type_df,
                  'plate_df': plate_df,
                  'reservoir_tag': reservoir_tag,
                  'ignore_tag': assay_plate_tag,
                  'tip_size': tip_size,
                  'n_per_group': 8,
                  'nzfill': 4}
    output = make_solution_worklist(source_unique, **input_dict)

    if output['worklist'].shape[0] > 0:
        sol_worklist = output['worklist'].copy()
        run_worklist, sol_worklist = shift_plate_worklist(run_worklist, sol_worklist)
        run_worklist = match_from_to_imaging(run_worklist)

        link_worklist = get_link_sol_run(sol_worklist, run_worklist, tip_size=tip_size, n_per_group=n_per_group)

        worklist_combo = worklist_concat(sol_worklist, link_worklist)
        worklist_combo = worklist_concat(worklist_combo, run_worklist)
        worklist = consolidate_transfer(worklist_combo, keep_tag=assay_plate_tag, reservoir_tag=reservoir_tag)
        worklist = update_holdover_volume_plate_tip(worklist, plate_df, nzfill, assay_plate_tag, reservoir_tag, tip_size)
        worklist = update_dispense_type(worklist, ignore_tag=assay_plate_tag, reservoir_tag=reservoir_tag)
        worklist = squeeze_plate_index(worklist, nzfill=nzfill)
    else:
        worklist = run_worklist

    user_solution = solution_user_input(worklist, plate_df, 'source', reservoir_tag)
    user_labware = get_labware(worklist, reservoir_tag)
    user_tip = get_tip_count(worklist)

    return {'worklist': worklist,
            'user_solution': user_solution,
            'user_labware': user_labware,
            'user_tip': user_tip}


def update_liquid_class(worklist_input, liquid_type_df_input):
    """
    update liquid classes
    :param worklist_input: worklist
    :param liquid_type_df_input: liquid type
    :return: new worklist
    """
    worklist = worklist_input.copy()
    worklist['liquid_type'] = worklist['liquid_class'].str.split('_', expand=True).iloc[:, 2]
    liquid_type_df = liquid_type_df_input.copy()
    liquid_type_df.columns = ['source', 'liquid_type']

    worklist = worklist.merge(liquid_type_df, on='source', how='left')
    worklist['liquid_type'] = worklist['liquid_type_y'].fillna(worklist['liquid_type_x'])
    worklist.drop(['liquid_type_y', 'liquid_type_x'], axis=1)
    worklist['liquid_class'] = 'ivl_tip' + \
                               worklist['tip_type'].astype(int).astype(str) + '_' + \
                               worklist['liquid_type'] + '_' + \
                               worklist['dispense_type'].str.replace('[^a-zA-Z]+', '')
    return(worklist)
