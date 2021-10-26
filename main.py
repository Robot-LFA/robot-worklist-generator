from util import *
from full_experiment import make_worklist_full_2d
import pandas as pd
import glob
import os
import inspect


def main():
    input_df = pd.read_csv('input_master.csv')
    dict_val = pd.to_numeric(input_df['value'], errors='coerce').astype('Int64').astype('object')
    dict_val.loc[dict_val.isna()] = input_df.loc[dict_val.isna(), 'value']
    input_dict = dict(zip(input_df['key'], dict_val))

    input_dict['exp_input'] = pd.read_csv(input_dict['exp_input_file'])
    input_dict['plate_df'] = pd.read_csv(input_dict['plate_df_file'])
    input_dict['time_df'] = pd.read_csv(input_dict['time_df_file'])
    input_dict['sol_df'] = get_sol_df(input_dict['sol_df_file'])
    input_dict['liquid_type_df'] = pd.read_csv(input_dict['liquid_type_df_file'])
    input_dict['tip_size'] = np.array(input_dict['tip_size_string'].split(',')).astype(int)
    input_dict['n_per_group'] = input_dict['npergroup']

    full_dir = input_dict['full_dir']

    # make directories
    for each_dir in [input_dict['output_dir'], input_dict['full_dir']]:
        if not os.path.exists(each_dir):
            os.makedirs(each_dir)

    # make assay worklists
    keys = list(inspect.signature(make_worklist_full_2d).parameters.keys())
    values = [input_dict[each] for each in keys]
    current_input = dict(zip(keys, values))
    make_worklist_full_2d(**current_input)

    # make full worklists
    keys = list(inspect.signature(full_from_run_worklist).parameters.keys())
    keys = np.setdiff1d(keys, 'run_worklist_input')
    values = [input_dict[each] for each in keys]
    current_input = dict(zip(keys, values))

    run_worklist_files = glob.glob(os.path.join(input_dict['output_dir'], '*_worklist.csv'))

    for each_file in run_worklist_files:
        run_worklist = pd.read_csv(each_file)
        full = full_from_run_worklist(run_worklist, **current_input)

        # update liquid class
        full['worklist'] = update_liquid_class(full['worklist'], input_dict['liquid_type_df'])

        base_name = os.path.basename(each_file).replace('worklist.csv', '')
        full['worklist'].to_csv(os.path.join(full_dir, base_name + 'full_worklist.csv'), index=False)
        full['user_solution'].to_csv(os.path.join(full_dir, base_name + 'full_user_solution.csv'), index=False)
        full['user_labware'].to_csv(os.path.join(full_dir, base_name + 'full_user_labware.csv'), index=False)
        full['user_tip'].to_csv(os.path.join(full_dir, base_name + 'full_user_tip.csv'), index=False)

main()
# if __name__ == 'main':
#      main()

