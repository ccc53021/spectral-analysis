from gurobipy import *
import numpy as np

import helper.inequalities as inequalities
from helper.skinny import skinny
import helper.utils as utils
import helper.printing as printing
import helper.linear_algebra as linear_algebra

class skinny_milp:
    def __init__(self,nr,tk,diff_characteristic,solution_set=None):
        if solution_set is None:
            solution_set = []
        self.model = Model('skinny_quasi')
        self.model.Params.OutputFlag = 0
        self.model.Params.Threads = utils.THREADS
        self.model.Params.LazyConstraints = 1
        self.model.Params.TimeLimit = utils.MAX_SEARCH_TIME  # seconds

        # need to define all the variables
        self.nr = nr
        self.sbox_size = utils.SBOX_SIZE
        self.ks_perm = skinny.ks_perm
        if tk == 0:
            self.key_size = 1
        else:
            self.key_size = tk

        if self.sbox_size == 4:
            self.corr_range = [0.0, 3.0, 2.0, 1.0]

        if self.sbox_size == 8:
            self.corr_range = [0.0, 5.0, 2.0, 1.0, 3.41504, 3.0, 4.0, 2.41504, 1.83007, 4.41504, 1.09311, 2.19265, 1.41504, 1.29956, 1.54057, 3.67807, 2.67807, 6.0, 3.19265, 7.00009, 5.41501]
        self.solution_count = 0
        self.diff_char = diff_characteristic
        self.dict_ineqs = {}
        self.solution_set = solution_set
        if utils.VERBOSE:
            print('Generating inequalities...')
        self.generate_ineqs() # this sets the combined correlation values as well


        self.state_mask_vars = self.model.addVars(self.nr,3,4,4,self.sbox_size,vtype=GRB.BINARY,name='state_mask')
        self.keys_mask_vars = self.model.addVars(self.nr,self.key_size,4,4,self.sbox_size,vtype=GRB.BINARY,name='key_mask')
        self.keys_mask_sum_vars = self.model.addVars(self.key_size,4,4,self.sbox_size,vtype=GRB.INTEGER,name='key_mask_sum')
        self.keys_mask_quotient_vars = self.model.addVars(self.key_size,4,4,self.sbox_size,vtype=GRB.INTEGER,name='key_mask_quotient')
        self.keys_mask_mod2_vars = self.model.addVars(self.key_size,4,4,self.sbox_size,vtype=GRB.BINARY,name='master_key')
        
        self.state_corr_vars = self.model.addVars(self.nr,4,4,self.corr_range,vtype=GRB.BINARY,name='state_corr')
        self.objective_function = self.model.addVar(lb=0,vtype=GRB.CONTINUOUS,name='obj_fun')


        # avoid all zero solution:
        self.model.addConstr(self.state_mask_vars.sum() >= 1)

        x_vars = utils.flatten_dict(self.keys_mask_vars) + utils.flatten_dict(self.state_mask_vars)
        found_solutions = []
        for sol in solution_set:
            format_key = utils.flatten(utils.dict_to_array(sol.keys_mask_vars))
            format_state = utils.flatten(utils.dict_to_array(sol.state_mask_vars))
            found_solutions.append(format_key + format_state)

        self.x_vars = x_vars
        self.found_solutions = found_solutions
        self.exclude_span_cut(self.x_vars, self.found_solutions)


    def exclude_span_cut(self, x_vars, found_vectors, lazy=False):
        # find the nullspace and let the search happens in the nullspace of the found vectors
        n = len(x_vars)
        checks = linear_algebra.gf2_nullspace_basis(found_vectors, n)

        parity_vars = []
        for j, h in enumerate(checks):
            support = [i for i in range(n) if h[i] == 1]
            if not support:
                continue
            p = self.model.addVar(vtype=GRB.BINARY, name=f"parity_{j}")
            t = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=len(support)//2, name=f"carry_{j}")
            self.model.addConstr(quicksum(x_vars[i] for i in support) == p + 2 * t)
            parity_vars.append(p)
        if lazy:
            self.model.cbLazy(quicksum(parity_vars) >= 1)    
        else:
            self.model.addConstr(quicksum(parity_vars) >= 1)

            
    def init(self):
        # return
        # self.model.addConstrs(self.state_mask_vars[0,0,r,c,i] == 0 for r in range(4) for c in range(4) for i in range(self.sbox_size))
        self.model.addConstrs(self.state_mask_vars[self.nr-1,2,r,c,i] == 0 for r in range(4) for c in range(4) for i in range(self.sbox_size))

    def generate_ineqs(self):
        for n in range(self.nr):
            for r in range(4):
                for c in range(4):
                    diff_in = self.diff_char.rounds[n].beforeSB[4*r+c]
                    diff_out = self.diff_char.rounds[n].afterSB[4*r+c]
                    if f'sbox_{diff_in}_{diff_out}' not in self.dict_ineqs.keys():
                        if utils.VERBOSE:
                            print(f'Generating sbox_{diff_in}_{diff_out}')
                        T = skinny.get_quasidifferential_sbox_submatrix_by_diff(diff_out,diff_in)
                        self.dict_ineqs[f'sbox_{diff_in}_{diff_out}'] = {}
                        for corr in self.corr_range:
                            data = inequalities.construct_inequalities(T,f'inequalities/sbox{self.sbox_size}/sbox_{diff_in}_{diff_out}_{corr}',corr)
                            if data is None: continue
                            self.dict_ineqs[f'sbox_{diff_in}_{diff_out}'][corr] = data
                    

    def sbox(self,my,mx,corrs,dict_ineqs,name=''):
        for corr in self.corr_range:
            if corr not in dict_ineqs.keys():
                self.model.addConstr(corrs[self.corr_range.index(corr)] == 0,name = name+'sbox=0')
                continue
            for ineq in dict_ineqs[corr].ineqs:
                self.model.addConstr((corrs[self.corr_range.index(corr)] == 1) >> 
                    (quicksum((1 - mask) if coeff == '1' else 0 if coeff == '-' else mask for
                coeff,mask in zip(ineq,[my[i] for i in range(self.sbox_size)] + 
                                        [mx[i] for i in range(self.sbox_size)])) >= 1),name=name+f'_{corr}'
                )
        return         
        
    def substitution(self,n):
        for r in range(4):
            for c in range(4):
                diff_in = self.diff_char.rounds[n].beforeSB[4*r+c]
                diff_out = self.diff_char.rounds[n].afterSB[4*r+c]
                ineqs = self.dict_ineqs[f'sbox_{diff_in}_{diff_out}']
                mx = [self.state_mask_vars[n,0,r,c,i] for i in range(self.sbox_size)]
                my = [self.state_mask_vars[n,1,r,c,i] for i in range(self.sbox_size)]
                corrs = [self.state_corr_vars[n,r,c,i] for i in self.corr_range]
                self.sbox(my,mx,corrs,ineqs,name=f'sbox_{diff_in}_{diff_out}')
                self.model.addConstr(quicksum(corrs) == 1,name=f'sbox_{diff_in}_{diff_out}_sbox_sum')

    def xor3(self,w,x,y,z):
        self.model.addConstr(w + (1-x) + (1-y) + (1-z) >= 1)
        self.model.addConstr((1-w) + x + (1-y) + (1-z) >= 1)
        self.model.addConstr((1-w) + (1-x) + y + (1-z) >= 1)
        self.model.addConstr((1-w) + (1-x) + (1-y) + z >= 1)

        self.model.addConstr((1-w) + x + y + z >= 1)
        self.model.addConstr(w + (1-x) + y + z >= 1)
        self.model.addConstr(w + x + (1-y) + z >= 1)
        self.model.addConstr(w + x + y + (1-z) >= 1)
        
    def add_round_key_cell(self,mx,my,mk,lfsr_counter):#,dx,dy,dk):
        # mask
        self.model.addConstrs(my[i] == mx[i] for i in range(self.sbox_size))

        # induce mask restrictions.
        if self.sbox_size == 4:
            ineq2_file = f'./inequalities/LFSR2S/count_{lfsr_counter}_after_espresso.pla'
            ineq3_file = f'./inequalities/LFSR3S/count_{lfsr_counter}_after_espresso.pla'
        elif self.sbox_size == 8:
            ineq2_file = f'./inequalities/LFSR2L/count_{lfsr_counter}_after_espresso.pla'
            ineq3_file = f'./inequalities/LFSR3L/count_{lfsr_counter}_after_espresso.pla'

        # for TK1 no changes
        self.model.addConstrs(my[i] == mk[0][i] for i in range(self.sbox_size))

        if self.key_size >= 2:
            ineq2s = inequalities.read_espresso(ineq2_file)
            for ineq in ineq2s:
                self.model.addConstr(quicksum((1 - mask) if coeff == '1' else 0 if coeff == '-' else mask for
                    coeff,mask in zip(ineq,[mk[1][i] for i in range(self.sbox_size)] + [my[i] for i in range(self.sbox_size)]
                                            )) >= 1,name='ark_cell1')
        if self.key_size >= 3:
            ineq3s = inequalities.read_espresso(ineq3_file)
            for ineq in ineq3s:
                self.model.addConstr(quicksum((1 - mask) if coeff == '1' else 0 if coeff == '-' else mask for
                    coeff,mask in zip(ineq,[mk[2][i] for i in range(self.sbox_size)] + [my[i] for i in range(self.sbox_size)]
                                             )) >= 1,name='ark_cell2')
        return

    def add_round_key(self,n):
        for r in range(4):
            for c in range(4):
                mx = [self.state_mask_vars[n,1,r,c,i] for i in range(self.sbox_size)]
                my = [self.state_mask_vars[n,2,r,c,i] for i in range(self.sbox_size)]
                tn,tr,tc = n,r,c
                lfsr_counter = 0
                while tn > 0: 
                    if tr <= 1: lfsr_counter += 1
                    tr,tc = skinny.invperm(tr,tc)
                    tn -= 1
                if utils.VERBOSE:
                    print(f'n is {n} lfsr_counter is {lfsr_counter}')
                if r < 2:
                    mk = [[self.keys_mask_vars[n,i,tr,tc,j] for j in range(self.sbox_size)] for i in range(self.key_size)]
                    self.add_round_key_cell(mx,my,mk,lfsr_counter)
                else:
                    self.model.addConstrs(my[i] == mx[i] for i in range(self.sbox_size))
                    self.model.addConstrs(self.keys_mask_vars[n,i,tr,tc,j] == 0 for i in range(self.key_size) for j in range(self.sbox_size))
    
    def linear_column(self,mx,my):
        self.xor3(mx[0],my[0],my[1],my[3])
        self.model.addConstr(mx[1] == my[2])
        self.xor3(mx[2],my[0],my[2],my[3])
        self.model.addConstr(mx[3] == my[0])
    
    def linear(self,n):
        for c in range(4):
            for bit in range(self.sbox_size):
                mx = [self.state_mask_vars[n,2,r,(c-r)%4,bit] for r in range(4)]
                my = [self.state_mask_vars[n+1,0,r,c,bit] for r in range(4)]
                self.linear_column(mx,my)
    
    def set_objective_function(self):
        self.model.addConstr(self.objective_function == quicksum(self.state_corr_vars[n,i,j,c] * c for n in range(self.nr) for i in range(4) for j in range(4) for c in self.corr_range))

    def run(self,max_trails):
        # self.model.write('skinny.lp')
        # add in max_correlation if it has
        if utils.MAX_CORRELATION is not None:
            self.model.addConstr(self.objective_function <= utils.MAX_CORRELATION)

        # Running the model
        self.model.setObjective(self.objective_function,GRB.MINIMIZE)

        if utils.LINEAR_TRAILS_ONLY:
            self.model.params.PoolSearchMode = 1
            self.model.params.PoolSolutions = 1
            self.model.optimize()
        elif not utils.LINEAR_TRAILS_ONLY:
            self.model.params.PoolSearchMode = 2
            self.model.params.PoolSolutions = max_trails
            self.model.optimize()
        
        # just need the status of it
        if self.model.Status == GRB.TIME_LIMIT:
            print("Terminated because of time limit")

        elif self.model.Status == GRB.INFEASIBLE:
            print("Model is infeasible")
            # for debugging purposes only
            # self.model.computeIIS()
            # self.print_infeasible()

        elif self.model.Status == GRB.OPTIMAL:
            print("Optimal solution found")

        # get the optimal solution
        if self.model.Status == GRB.OPTIMAL:
            n_solutions = self.model.SolCount
            for sol in range(n_solutions):
                self.model.Params.SolutionNumber = sol
                state_all_mask_int_vars = int(''.join(str(int(self.state_mask_vars[n,i,j,k,l].Xn)) for n in range(self.nr) for i in range(3) for j in range(4) for k in range(4) for l in range(self.sbox_size)),2)      
                state_mask_int_vars = int(''.join(str(int(self.state_mask_vars[0,0,j,k,l].Xn)) for j in range(4) for k in range(4) for l in range(self.sbox_size)),2)      
                keys_mask_mod2_int_vars = int(''.join(str(int(self.keys_mask_mod2_vars[i, j, k, l].Xn))for i in range(self.key_size)for j in range(4)for k in range(4)for l in range(self.sbox_size)),2)

                state_mask_vars = {(n,i,j,k,l) : int(self.state_mask_vars[n,i,j,k,l].Xn) for n in range(self.nr) for i in range(3) for j in range(4) for k in range(4) for l in range(self.sbox_size)}
                keys_mask_vars = {(n,i,j,k,l) : int(self.keys_mask_vars[n,i,j,k,l].Xn) for n in range(self.nr) for i in range(self.key_size) for j in range(4) for k in range(4) for l in range(self.sbox_size)}
                keys_mask_sum_vars = {(i,j,k,l) : int(self.keys_mask_sum_vars[i,j,k,l].Xn) for i in range(self.key_size) for j in range(4) for k in range(4) for l in range(self.sbox_size)}
                keys_mask_quotient_vars = {(i,j,k,l) : int(self.keys_mask_quotient_vars[i,j,k,l].Xn) for i in range(self.key_size) for j in range(4) for k in range(4) for l in range(self.sbox_size)}
                keys_mask_mod2_vars = {(i,j,k,l) : int(self.keys_mask_mod2_vars[i,j,k,l].Xn) for i in range(self.key_size) for j in range(4) for k in range(4) for l in range(self.sbox_size)}
                state_corr_vars = {(n,i,j,k) : float(self.state_corr_vars[n,i,j,k].Xn) for n in range(self.nr) for i in range(4) for j in range(4) for k in self.corr_range}

                objective_value = float(self.objective_function.Xn)
                sign,corr = skinny.get_correlation(self.diff_char,state_mask_vars)
                solution = utils.Solution(objective_value,sign,state_mask_int_vars,state_all_mask_int_vars,keys_mask_mod2_int_vars,state_mask_vars,keys_mask_vars,keys_mask_sum_vars,keys_mask_quotient_vars,keys_mask_mod2_vars,state_corr_vars)
                self.solution_set.append(solution)

                # printing.print_stats(self,state_mask_vars,keys_mask_vars,state_corr_vars,objective_value,sign)
                # printing.print_mask_trail(self,state_mask_vars,keys_mask_vars,state_corr_vars)
            return self.solution_set,1
        else:
            return self.solution_set,0

    def sum_mask(self):
        for k in range(self.key_size):
            for r in range(4):
                for c in range(4):
                    for b in range(self.sbox_size):
                        self.model.addConstr(self.keys_mask_sum_vars[k,r,c,b] == quicksum(self.keys_mask_vars[n,k,r,c,b] for n in range(self.nr)))
        return
    
    def mod2_mask(self):
        for k in range(self.key_size):
            for r in range(4):
                for c in range(4):
                    for b in range(self.sbox_size):
                        self.model.addConstr(self.keys_mask_sum_vars[k,r,c,b] == 2 * self.keys_mask_quotient_vars[k,r,c,b] + self.keys_mask_mod2_vars[k,r,c,b])
        return
    
    def print_infeasible(self):
        print('Conflicting constraints:')
        for constr in self.model.getConstrs():
            if constr.IISConstr:
                print(f"  ✗ {constr.ConstrName}")
        print("Conflicting variable bounds:")
        for var in self.model.getVars():
            if var.IISLB:
                print(f"  ✗ {var.VarName} lower bound: {var.LB}")
            if var.IISUB:
                print(f"  ✗ {var.VarName} upper bound: {var.UB}")

        for constr in self.model.getConstrs():
            if constr.IISConstr:
                # Get constraint details
                row = self.model.getRow(constr)
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
        assert False
    
        




        

    
