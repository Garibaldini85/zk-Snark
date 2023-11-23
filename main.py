class QAP:
    def __init__(self):
        self.dict_plus_gates = {}
        self.dict_plus_gates_rev = {}
        self.dict_mult_gates = {}
        self.dict_mult_gates_rev = {}
        self.list_qap = [1]
        
    def __str__(self):
        print(self.dict_plus_gates)
        print(self.dict_mult_gates)
        print(self.list_qap)
        return 'kek'
    
    def get_plus_gate(self, leafs, if_added=False):
        if self.is_exists_plus_gate(leafs[0], leafs[1]):
            return self.dict_plus_gates.get(f'{leafs[0]}:{leafs[1]}')
        
        if if_added:
            self.add_plus_gate(leafs[0], leafs[1])
            return self.dict_plus_gates.get(f'{leafs[0]}:{leafs[1]}')
        else:
            return None
        
    def get_mult_gate(self, leafs, if_added=False):
        if self.is_exists_mult_gate(leafs[0], leafs[1]):
            return self.dict_mult_gates.get(f'{leafs[0]}:{leafs[1]}')
        
        if if_added:
            self.add_mult_gate(leafs[0], leafs[1])
            return self.dict_mult_gates.get(f'{leafs[0]}:{leafs[1]}')
        else:
            return None
    
    def add_plus_gate(self, leaf_left, leaf_right):
        if self.is_exists_plus_gate(leaf_left, leaf_right):
            return
        self.dict_plus_gates.update({f'p_{len(self.dict_plus_gates)}': f'{leaf_left}:{leaf_right}'})
        self.dict_plus_gates_rev.update({f'{leaf_left}:{leaf_right}': f'p_{len(self.dict_plus_gates)}'})
        
    def add_mult_gate(self, leaf_left, leaf_right):
        if self.is_exists_mult_gate(leaf_left, leaf_right):
            return
        self.dict_plus_gates.update({f'm_{len(self.dict_plus_gates)}': f'{leaf_left}:{leaf_right}'})
        self.dict_plus_gates_rev.update({f'{leaf_left}:{leaf_right}': f'm_{len(self.dict_plus_gates)}'})

    def is_exists_plus_gate(self, leaf_left, leaf_right):
        return self.dict_plus_gates_rev.get(f'{leaf_left}:{leaf_right}') is not None
    
    def is_exists_mult_gate(self, leaf_left, leaf_right):
        return self.dict_mult_gates_rev.get(f'{leaf_left}:{leaf_right}') is not None
    
    def get_leafs_into_func_by_index(self, str_func, ind):
        leaf_left, leaf_right = '', ''
        for i in range(ind, 0, -1):
            if str_func[i-1] != '+' and str_func[i-1] != '*':
                leaf_left = str_func[i-1] + leaf_left
            else:
                break
        for i in range(ind, len(str_func) - 1):
            if str_func[i+1] != '+' and str_func[i+1] != '*':
                leaf_right =  leaf_right + str_func[i+1]
            else:
                break
        return (leaf_left, leaf_right)
    
    def init_qap(self, str_func):
        isTrue = True
        
        while isTrue:
            ind_mult = str_func.find('*')
            if ind_mult == -1:
                isTrue = False
            else:
                leafs = self.get_leafs_into_func_by_index(str_func, ind_mult)
                print(leafs)
                print(ind_mult - len(leafs[0]), ind_mult + len(leafs[1]) - len(leafs[0]) + 1)
                str_func = str_func.replace(leafs[0] + '*' + leafs[1], 'q', 1)
                # str_func = str_func[:ind_mult - len(mem[0])] + 'q' + str_func[ind_mult + len(mem[1]) - len(mem[0]) + 1:]
                print(str_func)
        return str_func
            
ban = QAP()
str_func = 'y_1*zqwer*ret'
print(str_func)
print(ban)
print(ban.init_qap(str_func))
