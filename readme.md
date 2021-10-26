# Robot worklist generator

This piece of software is used to generate worklist to run on IVL's Hamilton robots using the master methods available on them. It generate both worklists to run the assays alone and full worklists to make solutions, then run the assays. It also split a big experiment into smaller ones that fit on the deck. It deals with factorial experiments only at the moment, but worklists to run more customized experiments can be be made using some parts here and extra code.

Example files are already included.

Inputs:
* [input_master.csv](input_master.csv). Descriptions are also in here.
* [input_experiment](input_experiment). Note that in the input file such as [input_experiment/factorial_experiment.csv](input_experiment/factorial_experiment.csv), in the time column, -1 means the next step can happy any time after, 0 mean the next step has to happy immediately after (it's never 0 in reality), and a positive number specifies the time delay. 
* [input_instrument](input_instrument)
* [input_liquid](input_instrument)

Outputs:
* [output_run_assay_worklist](output_run_assay_worklist): worklists to run the assays only
* [output_full_worklist](output_full_worklist): full worklists