import argparse

from cipher_config import total_rounds, n, m, d_str, basis_number, weight_range, diffs, input_x
from cipher_config import characteristic_model_number as route_number


# =====================================================================================
# step 1. search quasidifferential bases
# =====================================================================================

from datetime import datetime
import quasidifferential_search
import utils

def step_1_quasidifferential_search(characteristic_limit=None):
    str_split = "-" * 100

    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]")
    print(f"Step 1. Get quasidifferentials, basis number = {basis_number}")

    limit = len(diffs) if characteristic_limit is None else min(characteristic_limit, len(diffs))
    for idx in range(limit):
        characteristic = diffs[idx]
        print(len(characteristic) // route_number)
        if (len(characteristic) // route_number) != total_rounds:
            print(f"error, total rounds = {total_rounds}, route rounds = {len(characteristic) // route_number}")
            exit()

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}]")
        print(f"DC[{idx}] : Get quasidifferentials:")
        print("Searching ... ")

        average_w = utils.get_average_w(characteristic, n, m)
        min_w = average_w
        max_w = min_w + weight_range
        file_name = f"output/record_quasidifferentials/step1_quasidc_search_r_{total_rounds}_{d_str}_c_{idx}_x_{input_x}_basis_number_{basis_number}_w_{min_w}_to_{max_w}.jsonl"
        quasidifferential_bases, w_list, positive_list, negative_list = quasidifferential_search.quasi_differential_search(characteristic, average_w, min_w, max_w, basis_number, input_x, file_name)
        for i in range(len(w_list)):
            print(f"DC[{idx}] : w = {w_list[i]}, all = {positive_list[i] + negative_list[i]}, sign + = {positive_list[i]}, sign - = {negative_list[i]}")

        print(str_split)
        print()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}]")
    print(str_split)
    print()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--characteristic-limit", type=int, default=None)
    args = parser.parse_args()
    if args.characteristic_limit is not None and args.characteristic_limit <= 0:
        parser.error("--characteristic-limit must be positive")
    return args


if __name__ == "__main__":
    step_1_quasidifferential_search(parse_args().characteristic_limit)
