from src.experiment_types import StaticBendExp, StaticTwistExp, StaticBendTetwiseExp, StaticTwistTetwiseExp, DynamicBendExp


lh_e = StaticBendExp(exp_file=['./experiments/paper_experiments/lh_e/lh_e_elasticity.exp'],
                    initial_parameters=[5e4, 2.5e6, 5.],
                    perturb_type='large',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e6,
                    background_threshold=[0.520],
                    device='cuda',
                    set_seed=True)

spine_e = StaticBendExp(exp_file=['./experiments/paper_experiments/spine_e/spine_e_elasticity.exp'],
                    initial_parameters=[5e4, 2.5e6, 5.],
                    perturb_type='small',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e3,
                    background_threshold=[0.43352],
                    device='cuda',
                    set_seed=True)

lh_m = StaticBendExp(exp_file=['./experiments/paper_experiments/lh_m/lh_m_elasticity.exp'],
                    initial_parameters=[5e4, 2.5e6, 5.],
                    perturb_type='large',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e6,
                    background_threshold=[0.60694],
                    device='cuda',
                    set_seed=True)

spine_m = StaticBendExp(exp_file=['./experiments/paper_experiments/spine_m/spine_m_elasticity.exp'],
                    initial_parameters=[4e4, 2.5e6, 5.],
                    perturb_type='large',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e3,
                    background_threshold=[0.5780],
                    view=[1],
                    device='cuda',
                    set_seed=True)

lh_me = StaticBendExp(exp_file=['./experiments/paper_experiments/lh_m+e/lh_m+e_elasticity.exp'],
                    initial_parameters=[6e4, 2.5e6, 5., 6e4, 2.5e6, 5.],
                    perturb_type='small',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e7,
                    background_threshold=[0.520],
                    view=[14],
                    device='cuda',
                    set_seed=True)

twist_em = StaticTwistExp(exp_file=['./experiments/paper_experiments/twist_e+m/twist_e+m_elasticity.exp'],
                    initial_parameters=[4.5e4, 2.5e6, 5., 4.5e4, 2.5e6, 5.],
                    perturb_type='small',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e3,
                    background_threshold=[0.523],
                    view=[1],
                    device='cuda',
                    set_seed=True)

lh_me_tetwise = StaticBendTetwiseExp(exp_file=['./experiments/paper_experiments/lh_m+e/lh_m+e_elasticity_tetwise.exp'],
                    initial_parameters=[6e4, 2.5e6, 5.],
                    perturb_type='small',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e4,
                    background_threshold=[0.520],
                    view=[14],
                    device='cuda',
                    set_seed=True)

twist_em_tetwise = StaticTwistTetwiseExp(exp_file=['./experiments/paper_experiments/twist_e+m/twist_e+m_elasticity_tetwise.exp'],
                    initial_parameters=[5e4, 2.5e6, 5.],
                    perturb_type='small',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e4,
                    background_threshold=[0.523],
                    view=[1],
                    device='cuda',
                    set_seed=True)

dragon_e = StaticBendExp(exp_file=['./experiments/paper_experiments/dragon_e/dragon_e_elasticity.exp'],
                    initial_parameters=[5e4, 2.5e6, 5.], #not sure about 5e4
                    perturb_type='large',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=5e3,
                    background_threshold=[0.549],
                    device='cuda',
                    set_seed=True)

lh_e_dynamic = DynamicBendExp(exp_file=['./experiments/paper_experiments/lh_m/lh_m_viscosity_1.exp'],
                    initial_parameters=[7e4, 2.5e6, 5.0],
                    perturb_type='small',
                    optimizable=[1, 0, 1, 1],
                    num_iters=10,
                    lr=1e3,
                    background_threshold=[0.549],
                    device='cuda',
                    set_seed=True)