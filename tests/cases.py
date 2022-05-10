from src.experiment_types import StaticHangExp, StaticTwistExp, StaticHangTetwiseExp, StaticTwistTetwiseExp, DynamicHangExp


h_e_c = StaticHangExp(exp_file=['exp_data/h_e_c/h_e_c.exp'],
                    initial_parameters=[5e4, 2.5e6, 5.],
                    perturb_type='large',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e6,
                    background_threshold=[0.520],
                    device='cuda',
                    set_seed=True)

h_e_s = StaticHangExp(exp_file=['exp_data/h_e_s/h_e_s.exp'],
                    initial_parameters=[5e4, 2.5e6, 5.],
                    perturb_type='small',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e3,
                    background_threshold=[0.43352],
                    device='cuda',
                    set_seed=True)

h_m_c = StaticHangExp(exp_file=['exp_data/h_m_c/h_m_c.exp'],
                    initial_parameters=[5e4, 2.5e6, 5.],
                    perturb_type='large',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e6,
                    background_threshold=[0.60694],
                    device='cuda',
                    set_seed=True)

h_m_s = StaticHangExp(exp_file=['exp_data/h_m_s/h_m_s.exp'],
                    initial_parameters=[4e4, 2.5e6, 5.],
                    perturb_type='large',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e3,
                    background_threshold=[0.5780],
                    view=[1],
                    device='cuda',
                    set_seed=True)

h_em_c = StaticHangExp(exp_file=['exp_data/h_em_c/h_em_c.exp'],
                    initial_parameters=[6e4, 2.5e6, 5., 6e4, 2.5e6, 5.],
                    perturb_type='small',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e7,
                    background_threshold=[0.520],
                    view=[14],
                    device='cuda',
                    set_seed=True)

t_em_c = StaticTwistExp(exp_file=['exp_data/t_em_c/t_em_c.exp'],
                    initial_parameters=[4.5e4, 2.5e6, 5., 4.5e4, 2.5e6, 5.],
                    perturb_type='small',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e3,
                    background_threshold=[0.523],
                    view=[1],
                    device='cuda',
                    set_seed=True)

h_em_c_tetwise = StaticHangTetwiseExp(exp_file=['exp_data/h_em_c_tetwise/h_em_c_tetwise.exp'],
                    initial_parameters=[6e4, 2.5e6, 5.],
                    perturb_type='small',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e4,
                    background_threshold=[0.520],
                    view=[14],
                    device='cuda',
                    set_seed=True)

t_em_c_tetwise = StaticTwistTetwiseExp(exp_file=['exp_data/t_em_c_tetwise/t_em_c_tetwise.exp'],
                    initial_parameters=[5e4, 2.5e6, 5.],
                    perturb_type='small',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=1e4,
                    background_threshold=[0.523],
                    view=[1],
                    device='cuda',
                    set_seed=True)

h_e_d = StaticHangExp(exp_file=['exp_data/h_e_d/h_e_d.exp'],
                    initial_parameters=[5e4, 2.5e6, 5.],
                    perturb_type='large',
                    optimizable=[0, 1, 0, 0],
                    num_iters=10,
                    lr=5e3,
                    background_threshold=[0.549],
                    device='cuda',
                    set_seed=True)

o_m_c = DynamicHangExp(exp_file=['exp_data/o_m_c_fine/o_m_c_fine.exp'],
                    initial_parameters=[7e4, 2.5e6, 5.0],
                    perturb_type='small',
                    optimizable=[1, 0, 1, 1],
                    num_iters=10,
                    lr=1e3,
                    background_threshold=[0.549],
                    device='cuda',
                    set_seed=True)