for i in {0..9}
    do
    python3 -m main --type static_bendtwist_tetwise --exp ./experiments/paper_experiments/lh_m+e_bendtwist/lh_m+e_elasticity_tetwise.exp ./experiments/paper_experiments/lh_m+e_bendtwist/twist_e+m_elasticity_tetwise.exp --lr 1e6 --num_iters 50 --p0 6e4 2.5e6 5.0 --opt 0 1 0 0 --bg_thr 0.4925 0.523 --view 8 1 --perturb small --set_seed 0
    done