import signac

project = signac.init_project()

statepoint_list = [
        {'chain_length':100, 'lj_cutoff':1.12},
        {'chain_length':100, 'lj_cutoff':2.5},
        {'chain_length':200, 'lj_cutoff':1.12},
        {'chain_length':200, 'lj_cutoff':2.5}
        ]

for statepoint in statepoint_list:
    job = project.open_job(statepoint).init()

