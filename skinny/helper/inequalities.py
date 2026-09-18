import numpy as np
import subprocess
from pathlib import Path
import pickle
from helper.skinny import skinny

class ineq_data:
    def __init__(self,ineq_file,want_points,elim_points,ineqs):
        self.ineq_file = ineq_file
        self.want_points = want_points
        self.elim_points = elim_points
        self.ineqs = ineqs


def get_corr_index(corr,corr_range):
    for index,c in enumerate(corr_range):
        if np.abs(corr-c) < 0.001:
            return index
    else:
        raise Exception(f'correlation {c} is not in corr_range!')

def construct_inequalities(T,name,corr):
    ineq_dump = name + '_dump.pkl'
    input_filename = name + '_before_espresso.pla'
    output_filename = name + '_after_espresso.pla'
    ineq_dump_file = Path(ineq_dump)
    if ineq_dump_file.exists():
        with open(ineq_dump,'rb') as f:
            data = pickle.load(f)   
        return data
    else:
        want_points = []
        elim_points = []
        table_log_size = int(np.log2(T.shape[0]))
        no_cols = 2 * table_log_size
        # compute want_points
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if np.abs(T[i,j]) <= 0.0001: continue
                elif np.abs(np.abs(T[i,j]) - 2**(-corr)) < 0.0001: 
                    want_points.append((i << table_log_size) + j)
        if want_points == []: 
            return None
        print(f'generating the file {input_filename}...',flush=True)
        for i in range(1 << no_cols):
            if i not in want_points:
                elim_points.append(i)
        
        construct_espresso_file(elim_points,input_filename,table_log_size)
        print('running espresso...',flush=True)
        run_espresso(input_filename,output_filename)
        print('espresso completed...',flush=True)
        ineqs = read_espresso(output_filename)
        data = ineq_data(output_filename,want_points,elim_points,ineqs)
        with open(ineq_dump_file,'wb') as f:
            pickle.dump(data, f)
        return data


def read_espresso(filename):
    lines = []
    with open(filename,'r') as f:
        M = f.readlines()
        for line in M:
            if '.' == line[0]: continue
            line = list(line.split(' ')[0])
            lines.append([l for l in line])
    return lines



def construct_espresso_file(elim_points,input_file,table_log_size):
    # note: mx0 is the MSB
    s = '.ilb '
    for i in range(table_log_size):
        s += f'dy{i} '
    for i in range(table_log_size):
        s += f'dx{i} '

    with open(input_file,'w') as f:
        print(f'.i {2*table_log_size}',file=f)
        print(f'.o 1',file=f)
        print(f'.typefd',file=f)
        print(s,file=f)
        print(f'.ob INV',file=f)
        for point in elim_points:
            print(bin(point)[2:].zfill(2*table_log_size) + ' 1',file=f)
        print('.e',file=f)

def run_espresso(input_filename,output_filename):
    subprocess.run(
        f"espresso {input_filename} > {output_filename}",
        shell=True,
        check=True
    )

def convert_bit_array(x,size=4):
    return [1 if ((x >> (size-i-1)) & 1) else 0 for i in range(size)]
        
def compute_LFSR2S():
    LFSR2 = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0]],dtype=int) # LFSR
    # x = LFSR^T(y)
    max_count = 15
    for count in range(max_count):
        input_filename = f'./inequalities/LFSR2S/count_{count}_before_espresso.pla'
        output_filename = f'./inequalities/LFSR2S/count_{count}_after_espresso.pla'
        if Path(output_filename).exists():
            continue
        want_points = []
        for y in range(2**skinny.sbox_size):
            y_array = convert_bit_array(y,skinny.sbox_size)
            x_array = convert_bit_array(y,skinny.sbox_size)
            for _ in range(count): x_array = LFSR2.T.dot(x_array) % 2
            want_points.append(int(''.join([str(x_array[i]) for i in range(skinny.sbox_size)]) + ''.join([str(y_array[i]) for i in range(skinny.sbox_size)]),2))
        elim_points = []
        for i in range(2**(2*skinny.sbox_size)):
            if i not in want_points:
                elim_points.append(i)

        # note: mx0 is the MSB
        s = '.ilb '
        for i in range(skinny.sbox_size):
            s += f'x{i} '
        for i in range(skinny.sbox_size):
            s += f'y{i} '
        with open(input_filename,'w') as f:
            print(f'.i {2*skinny.sbox_size}',file=f)
            print(f'.o 1',file=f)
            print(f'.typefd',file=f)
            print(s,file=f)
            print(f'.ob INV',file=f)
            for point in elim_points:
                print(bin(point)[2:].zfill(2*skinny.sbox_size) + ' 1',file=f)
            print('.e',file=f)
        run_espresso(input_filename,output_filename)

def compute_LFSR3S():
    LFSR3 = np.array([[1,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=int) # LFSR
    # x = LFSR^T(y)
    max_count = 15
    for count in range(max_count):
        input_filename = f'./inequalities/LFSR3S/count_{count}_before_espresso.pla'
        output_filename = f'./inequalities/LFSR3S/count_{count}_after_espresso.pla'
        if Path(output_filename).exists():
            continue
        want_points = []
        for y in range(2**skinny.sbox_size):
            y_array = convert_bit_array(y,skinny.sbox_size)
            x_array = convert_bit_array(y,skinny.sbox_size)
            for _ in range(count): x_array = LFSR3.T.dot(x_array) % 2
            want_points.append(int(''.join([str(x_array[i]) for i in range(skinny.sbox_size)]) + ''.join([str(y_array[i]) for i in range(skinny.sbox_size)]),2))
        elim_points = []
        for i in range(2**(2*skinny.sbox_size)):
            if i not in want_points:
                elim_points.append(i)

        # note: mx0 is the MSB
        s = '.ilb '
        for i in range(skinny.sbox_size):
            s += f'x{i} '
        for i in range(skinny.sbox_size):
            s += f'y{i} '
        with open(input_filename,'w') as f:
            print(f'.i {2*skinny.sbox_size}',file=f)
            print(f'.o 1',file=f)
            print(f'.typefd',file=f)
            print(s,file=f)
            print(f'.ob INV',file=f)
            for point in elim_points:
                print(bin(point)[2:].zfill(2*skinny.sbox_size) + ' 1',file=f)
            print('.e',file=f)
        run_espresso(input_filename,output_filename)


def compute_LFSR2L():
    LFSR2 = np.array([[0,1,0,0, 0,0,0,0],
                      [0,0,1,0, 0,0,0,0],
                      [0,0,0,1, 0,0,0,0],
                      [0,0,0,0, 1,0,0,0],
                      [0,0,0,0, 0,1,0,0],
                      [0,0,0,0, 0,0,1,0],
                      [0,0,0,0, 0,0,0,1],
                      [1,0,1,0, 0,0,0,0],
                      ],dtype=int) # LFSR
    # x = LFSR^T(y)
    max_count = 15
    for count in range(max_count):
        input_filename = f'./inequalities/LFSR2L/count_{count}_before_espresso.pla'
        output_filename = f'./inequalities/LFSR2L/count_{count}_after_espresso.pla'
        if Path(output_filename).exists():
            continue
        want_points = []
        print(f'computing {input_filename}',flush=True)
        for y in range(2**skinny.sbox_size):
            y_array = convert_bit_array(y,skinny.sbox_size)
            x_array = convert_bit_array(y,skinny.sbox_size)
            for _ in range(count): x_array = LFSR2.T.dot(x_array) % 2
            want_points.append(int(''.join([str(x_array[i]) for i in range(skinny.sbox_size)]) + ''.join([str(y_array[i]) for i in range(skinny.sbox_size)]),2))
        elim_points = []
        for i in range(2**(2*skinny.sbox_size)):
            if i not in want_points:
                elim_points.append(i)

        # note: mx0 is the MSB
        s = '.ilb '
        for i in range(skinny.sbox_size):
            s += f'x{i} '
        for i in range(skinny.sbox_size):
            s += f'y{i} '
        with open(input_filename,'w') as f:
            print(f'.i {2*skinny.sbox_size}',file=f)
            print(f'.o 1',file=f)
            print(f'.typefd',file=f)
            print(s,file=f)
            print(f'.ob INV',file=f)
            for point in elim_points:
                print(bin(point)[2:].zfill(2*skinny.sbox_size) + ' 1',file=f)
            print('.e',file=f)
        print(f'computing {output_filename} ...')
        run_espresso(input_filename,output_filename)
        print(f'finished computing {output_filename} ...')

def compute_LFSR3L():
    LFSR3 = np.array([[0,1,0,0, 0,0,0,1],
                      [1,0,0,0, 0,0,0,0],
                      [0,1,0,0, 0,0,0,0],
                      [0,0,1,0, 0,0,0,0],
                      [0,0,0,1, 0,0,0,0],
                      [0,0,0,0, 1,0,0,0],
                      [0,0,0,0, 0,1,0,0],
                      [0,0,0,0, 0,0,1,0]],dtype=int) # LFSR
    # x = LFSR^T(y)
    max_count = 15
    for count in range(max_count):
        input_filename = f'./inequalities/LFSR3L/count_{count}_before_espresso.pla'
        output_filename = f'./inequalities/LFSR3L/count_{count}_after_espresso.pla'
        if Path(output_filename).exists():
            continue
        want_points = []
        for y in range(2**skinny.sbox_size):
            y_array = convert_bit_array(y,skinny.sbox_size)
            x_array = convert_bit_array(y,skinny.sbox_size)
            for _ in range(count): x_array = LFSR3.T.dot(x_array) % 2
            want_points.append(int(''.join([str(x_array[i]) for i in range(skinny.sbox_size)]) + ''.join([str(y_array[i]) for i in range(skinny.sbox_size)]),2))
        elim_points = []
        for i in range(2**(2*skinny.sbox_size)):
            if i not in want_points:
                elim_points.append(i)

        # note: mx0 is the MSB
        s = '.ilb '
        for i in range(skinny.sbox_size):
            s += f'x{i} '
        for i in range(skinny.sbox_size):
            s += f'y{i} '
        with open(input_filename,'w') as f:
            print(f'.i {2*skinny.sbox_size}',file=f)
            print(f'.o 1',file=f)
            print(f'.typefd',file=f)
            print(s,file=f)
            print(f'.ob INV',file=f)
            for point in elim_points:
                print(bin(point)[2:].zfill(2*skinny.sbox_size) + ' 1',file=f)
            print('.e',file=f)
        print(f'computing {output_filename} ...')
        run_espresso(input_filename,output_filename)
        print(f'finished computing {output_filename} ...')
