from stats import plot_errors_vs_H

folder = "logs"
black_list = ['T_synth']

runs = {
    50: ['good_50_5', 'RAN_50_3_ok'],
    100: ['RAN_100_5', 'RAN_100_7'],
    150: ['ran_150_16_good', 'ran_150_15_meh', 'ran_150_12_meh'],
    200: ['ANG_200_3', 'ANG_200_8', 'ANG_200_11', 'ANG_200_13', 'ANG_200_14', 'ANG_200_15'],
    500: ['G4_500_1', 'G4_500_4', 'G5_500_5'],
}


plot_errors_vs_H(runs, folder, black_list, average=True)
