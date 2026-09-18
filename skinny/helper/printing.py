import numpy as np
import helper.utils as utils

def print_trail(diff_char):
    for n in range(diff_char.nr):
        for r in range(4):
            for c in range(4):
                print(hex(diff_char.rounds[n].beforeSB[4*r+c])[2:],end='')
            print(' ',end='')
            for c in range(4):
                print(hex(diff_char.rounds[n].afterSB[4*r+c])[2:],end='')
            print(' ',end='')
            for c in range(4):
                print(hex(diff_char.rounds[n].afterARK[4*r+c])[2:],end='')
            print(' ',end='')
            for c in range(4):
                print(hex(diff_char.rounds[n].afterSR[4*r+c])[2:],end='')
            print(' ',end='')
            for c in range(4):
                print(hex(diff_char.rounds[n].afterMC[4*r+c])[2:],end='')
            print('|',end='')
            for k in range(diff_char.key_size):
                for c in range(4):
                    print(hex(diff_char.rounds[n].keyDiff[k][4*r+c])[2:],end='')
                print('|',end='')
            for c in range(4):
                print(int(diff_char.rounds[n].prob[4*r+c]),end='')
            print()
        print('-'*16)
    print('='*16)

def print_table(T,m=16,n=16):
    for i in range(m):
        for j in range(n):
            print(T[i][j],end=' ')
        print()
    print()

def print_mask_trail(instance,state_mask_vars,keys_mask_vars=None,state_corr_vars=None):
    for n in range(instance.nr):
        for r in range(4):
            for c in range(4):
                for i in range(instance.sbox_size):
                    try:
                        print(int(state_mask_vars[n][0][r][c][i].X),end='')
                    except:
                        print(int(state_mask_vars[n][0][r][c][i]),end='')
                print(' ',end='')
            print(' ',end='')
            for c in range(4):
                for i in range(instance.sbox_size):
                    try:
                        print(int(state_mask_vars[n][1][r][c][i].X),end='')
                    except:
                        print(int(state_mask_vars[n][1][r][c][i]),end='')
                print(' ',end='')
            print(' ',end='')
            for c in range(4):
                for i in range(instance.sbox_size):
                    try:
                        print(int(state_mask_vars[n][2][r][c][i].X),end='')
                    except:
                        print(int(state_mask_vars[n][2][r][c][i]),end='')
                print(' ',end='')
            print(' ',end='')
            for c in range(4):
                for i in range(instance.sbox_size):
                    try:
                        print(int(state_mask_vars[n][2][r][(c-r)%4][i].X),end='')
                    except:
                        print(int(state_mask_vars[n][2][r][(c-r)%4][i]),end='')
                print(' ',end='')
            print(' ',end='')
            if n < instance.nr-1:
                for c in range(4):
                    for i in range(instance.sbox_size):
                        try:
                            print(int(state_mask_vars[n+1][0][r][c][i].X),end='')
                        except:
                            print(int(state_mask_vars[n+1][0][r][c][i]),end='')
                    print(' ',end='')
            print('|',end='')
            if keys_mask_vars is not None:
                for t in range(instance.key_size):
                    for c in range(4):
                        for i in range(instance.sbox_size):
                            try:
                                print(int(keys_mask_vars[n][t][r][c][i].X),end='')
                            except:
                                print(int(keys_mask_vars[n][t][r][c][i]),end='')
                        print(' ',end='')
                print('|',end='')
            if state_corr_vars is not None:
                for c in range(4):
                    for index,i in enumerate(instance.corr_range):
                        try:
                            if int(state_corr_vars[n][r][c][i].X) == 1:
                                print(f'{np.abs(i):.1f}',end=' ')
                        except:
                            if int(state_corr_vars[n][r][c][index]) == 1:
                                print(f'{np.abs(instance.corr_range[index]):.1f}',end=' ')
            print()
        print('-'*16)
    print('='*16)

def print_infeasible(instance):
        print('Conflicting constraints:')
        for constr in instance.model.getConstrs():
            if constr.IISConstr:
                print(f"  ✗ {constr.ConstrName}")
        print("Conflicting variable bounds:")
        for var in instance.model.getVars():
            if var.IISLB:
                print(f"  ✗ {var.VarName} lower bound: {var.LB}")
            if var.IISUB:
                print(f"  ✗ {var.VarName} upper bound: {var.UB}")

        for constr in instance.model.getConstrs():
            if constr.IISConstr:
                # Get constraint details
                row = instance.model.getRow(constr)
                sense = constr.Sense
                rhs = constr.RHS
                # Build the constraint expression
                expr_str = ""
                for i in range(row.size()):
                    coeff = row.getCoeff(i)
                    var = row.getVar(i)
                    if i > 0 and coeff >= 0:
                        expr_str += " + "
                    elif coeff < 0:
                        expr_str += " - "
                        coeff = abs(coeff)
                    else:
                        pass
                    expr_str += f"{coeff} {var.VarName}"
                
                print(f"  {expr_str} {sense}= {rhs} | {constr.ConstrName}")

def print_stats(instance,state_mask_vars,keys_mask_vars,state_corr_vars,objective_value,sign):
    print(f'Objective value: {objective_value}')
    print(f'sign: {sign}')
    print('Keys involved')
    for n in range(instance.nr):
        for r in range(2):
            for c in range(4):
                for b in range(instance.sbox_size):
                    try:
                        if int(state_mask_vars[n,1,r,c,b].X) == 1:
                            print(f'k_{n}_{r}_{c}_{b}')
                    except:
                        if int(state_mask_vars[n,1,r,c,b]) == 1:
                            print(f'k_{n}_{r}_{c}_{b}')
    
