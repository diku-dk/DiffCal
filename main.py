from src.experiment_types import StaticHangExp, StaticTwistExp, StaticHangTwistExp, StaticHangTetwiseExp,\
                                 StaticTwistTetwiseExp, StaticHangTwistTetwiseExp, DynamicHangExp, DynamicTwistExp
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run an experiment.')
    parser.add_argument('--type', type=str, help='experiment type')
    parser.add_argument('--exp', type=str, nargs='+', help='filepath to .exp file')
    parser.add_argument('--p0', type=float, nargs='+', help='initial parameters')
    parser.add_argument('--bg_thr', type=float, nargs='+', help='background threshold to zero out small values in camera')
    parser.add_argument('--view', type=int, nargs='+', default=[-1,-1], help='which camera image to use, by default uses the last one.')
    parser.add_argument('--lr', type=float, default=1e3, help='learning_rate')
    parser.add_argument('--num_iters', type=int, default=10, help='maximum number of times to update parameters')
    parser.add_argument('--perturb', type=str, default='large', help='how much to perturb initial parameters')
    parser.add_argument('--opt', type=int, nargs=4, default=[0,1,0,0], help='set 1 to optimize [density, mu, lambda, damping], 0 otherwise')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--set_seed', type=bool, default=False, help='sets all seeds to 0')
    args = parser.parse_args()


    setting = args.exp, args.p0, args.perturb, args.opt, args.num_iters, args.lr, args.bg_thr, args.view, args.device, args.set_seed
    experiment_factory = {'static_hang': StaticHangExp,
                        'static_twist': StaticTwistExp,
                        'static_hangtwist': StaticHangTwistExp,
                        'static_hang_tetwise': StaticHangTetwiseExp,
                        'static_twist_tetwise': StaticTwistTetwiseExp,
                        'static_hangtwist_tetwise': StaticHangTwistTetwiseExp,
                        'dynamic_hang': DynamicHangExp,
                        'dynamic_twist': DynamicTwistExp
                        }
    if args.type not in experiment_factory:
        raise ValueError(f'experment_type should be one of {list(experiment_factory.keys())}.')
    else:
        experiment_factory[args.type](*setting).run()