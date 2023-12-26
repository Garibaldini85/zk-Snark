import re

def format_polynomial(expression):
    ret_expression = expression

    for i in re.findall(r'((\w+)|(\w+_\d+))\^(\d+)?', expression):
        new_expression = i[1]
        for _ in range(int(i[3]) - 1):
            new_expression += '*' + i[1]
        ret_expression = ret_expression.replace(f'{i[1]}^{i[3]}', new_expression)
    return ret_expression

class QAP:
    def __init__(self, P, monom):
        self.dict_plus_gates = {}
        self.dict_plus_gates_rev = {}
        self.dict_mult_gates = {}
        self.dict_mult_gates_rev = {}
        self.dict_multipliers_with_value = {}
        self.dict_qap_with_value = {}
        self.dict_qap_leaf_values = {}
        self.dict_qap_index = {}
        
        self.list_multipliers = []
        self.list_qap = []
        self.list_value_multipliers = []
        
        self.field = P
        self.func_P = None
        self.monom = monom
        self.list_A = []
        self.list_B = []
        self.list_C = []
       
    def set_all_multipliers(self, str_func):
        
        pattern = r'(\w+)|(\w+_\d+)'
        counter = 0
        for i in re.findall(pattern, str_func):
            if self.dict_qap_index.get(i[0]) is None:
                self.dict_qap_index.update({i[0]: counter})
                counter += 1

        self.list_multipliers = list(self.dict_qap_index.keys())
        self.list_qap = self.list_multipliers.copy()
        count = 1
        for i in self.list_multipliers:
            if i.isdigit():
                self.list_value_multipliers += [int(i)]
            else:
                self.list_value_multipliers += [count]
                count += 1
        for i in range(0, len(self.list_multipliers)):
            self.dict_multipliers_with_value.update({self.list_multipliers[i]: self.list_value_multipliers[i]})
                
        
    def __str__(self):
        print('_'*101)
        print("|> QAP   ")
        print("| | multipliers_with_value >> ", self.dict_multipliers_with_value)
        #print("dict_qap_index >>\t\t", self.dict_qap_index)
        print("| | dict_plus_gates >>        ", self.dict_plus_gates)
        print("| | dict_mult_gates >>        ", self.dict_mult_gates)
        
        #print(self.list_multipliers)
        #print("list_qap >>\t\t\t", self.list_qap)
        print("| | dict_qap_with_value >>    ", self.dict_qap_with_value)
        print("| | func_P              >>    ", self.func_P)
        return '|'+'_'*100
    
    def get_plus_gate(self, leafs, if_added=False):
        if self.is_exists_plus_gate(leafs[0], leafs[1]):
            return self.dict_plus_gates_rev.get(f'{leafs[0]}:{leafs[1]}')
        
        if if_added:
            self.add_plus_gate(leafs[0], leafs[1])
            return self.dict_plus_gates_rev.get(f'{leafs[0]}:{leafs[1]}')
        else:
            return None
        
    def get_mult_gate(self, leafs, if_added=False):
        if self.is_exists_mult_gate(leafs[0], leafs[1]):
            return self.dict_mult_gates_rev.get(f'{leafs[0]}:{leafs[1]}')
        
        if if_added:
            self.add_mult_gate(leafs[0], leafs[1])
            return self.dict_mult_gates_rev.get(f'{leafs[0]}:{leafs[1]}')
        else:
            return None
    
    def add_plus_gate(self, leaf_left, leaf_right):
        if self.is_exists_plus_gate(leaf_left, leaf_right):
            return
        #p_ind = len(self.dict_plus_gates)
        # if self.dict_qap_index()
        self.dict_plus_gates.update({f'p_{len(self.dict_plus_gates)}': f'{leaf_left}:{leaf_right}'})
        self.dict_plus_gates_rev.update({f'{leaf_left}:{leaf_right}': f'p_{len(self.dict_plus_gates_rev)}'})
        #self.dict_qap_index.update({f'p_{p_ind}': len(self.dict_qap_index)})
        
    def add_mult_gate(self, leaf_left, leaf_right):
        if self.is_exists_mult_gate(leaf_left, leaf_right):
            return
        m_ind = len(self.dict_mult_gates)
        self.dict_mult_gates.update({f'm_{m_ind}': f'{leaf_left}:{leaf_right}'})
        self.dict_mult_gates_rev.update({f'{leaf_left}:{leaf_right}': f'm_{m_ind}'})
        self.dict_qap_index.update({f'm_{m_ind}': len(self.dict_qap_index)})

    def is_exists_plus_gate(self, leaf_left, leaf_right):
        return self.dict_plus_gates_rev.get(f'{leaf_left}:{leaf_right}') is not None
    
    def is_exists_mult_gate(self, leaf_left, leaf_right):
        return self.dict_mult_gates_rev.get(f'{leaf_left}:{leaf_right}') is not None
    
    def get_leafs_from_func_by_index(self, str_func, ind):
        leaf_left, leaf_right = '', ''
        for i in range(ind, 0, -1):
            if str_func[i-1] not in ['*', '+', '(', ')']:
                leaf_left = str_func[i-1] + leaf_left
            else:
                break
        for i in range(ind, len(str_func) - 1):
            if str_func[i+1] not in ['*', '+', '(', ')']:
                leaf_right =  leaf_right + str_func[i+1]
            else:
                break
        return (leaf_left, leaf_right)
    
    
    
    def rebuild_func(self, str_func, sign):
        ind_sign = str_func.find(sign)
        if ind_sign == -1:
            return None
        
        leafs = self.get_leafs_from_func_by_index(str_func, ind_sign)
        if sign == '*':
            mult = self.get_mult_gate(leafs, True)
            str_func = str_func.replace(leafs[0] + '*' + leafs[1], mult, 1)
            if mult not in self.list_qap:
                self.list_qap += [mult]
        else:
            plus = self.get_plus_gate(leafs, True)
            mult = self.get_mult_gate((plus, '1'), True)
            str_func = str_func.replace(leafs[0] + '+' + leafs[1], mult, 1)
            if mult not in self.list_qap:
                self.list_qap += [mult]

        return str_func
    
    def reduce_func(self, str_func, sign):
        isNone = 0
        if str_func.find(sign) == -1:
            return str_func
        
        while isNone is not None:
            ret = self.rebuild_func(str_func, sign)
            if ret is None:
                #if (sign == '+'):
                    #str_func += '*1'
                    #str_func = self.reduce_func(str_func, '*')
                return str_func
            str_func = ret
        
    def get_indexes_internal_func(self, str_func):
        indexes_expr = [-1, -1]
        isTrue = True
        while isTrue:
            temp = str_func.find('(', indexes_expr[0] + 1)
            if temp != -1:
                indexes_expr[0] = temp
            else:
                isTrue = False
        if indexes_expr[0] == -1:
            return None
        
        indexes_expr[1] = str_func.find(')', indexes_expr[0] + 1) + 1
        return indexes_expr
    
    def get_value_by_leaf(self, leaf):
        if leaf.isdigit():
            return self.field(int(leaf))
        
        if self.dict_multipliers_with_value.get(leaf):
            return self.field(self.dict_multipliers_with_value.get(leaf))
        
        if self.dict_plus_gates.get(leaf):
            leafs = self.dict_plus_gates.get(leaf).split(':')
            return self.field(self.get_value_by_leaf(leafs[0]) + self.get_value_by_leaf(leafs[1]))
        
        if self.dict_mult_gates.get(leaf):
            leafs = self.dict_mult_gates.get(leaf).split(':')
            return self.field(self.get_value_by_leaf(leafs[0]) * self.get_value_by_leaf(leafs[1]))
    
    def get_value_qap(self, lst_cord=None):
        if lst_cord is not None:
            for i in range(0, len(self.list_multipliers)):
                meaning = self.list_multipliers[i]
                self.dict_multipliers_with_value.update({meaning: meaning if meaning.isdigit() else lst_cord[i]})
                
        self.dict_qap_with_value = {}
        for i in range(0, len(self.list_qap)):
            self.dict_qap_with_value.update({self.list_qap[i]: self.get_value_by_leaf(self.list_qap[i])})
        
        return self.dict_qap_with_value

    def refresh_func_P(self):
        cord = vector(self.field, [1] + [i for i in self.dict_qap_with_value.values()])
        
        Ax = self.field(list(cord*self.list_A))
        Bx = self.field(list(cord*self.list_B))
        Cx = self.field(list(cord*self.list_C))
        
        self.func_P = Ax*Bx - Cx
        
    def build_qap(self):
        qap_with_value = list(self.dict_qap_with_value.keys())
        temp_s = ['1'] + [i for i in self.dict_qap_index.keys()]
        dict_temp_s = {}
        for i in temp_s:
            dict_temp_s.update({i: 0})

        for i in self.list_qap:
            if self.dict_multipliers_with_value.get(i) is None:
                mult = self.dict_mult_gates.get(i)
                leafs = mult.split(':')
                
                a = dict_temp_s.copy()
                b = dict_temp_s.copy()
                c = dict_temp_s.copy()
                
                if self.dict_plus_gates.get(mult.split(':')[0]) is not None:
                    plus = self.dict_plus_gates.get(leafs[0])
                    plus_leafs = plus.split(':')
                    
                    a.update({plus_leafs[0]: 1, plus_leafs[1]: 1})
                    b.update({leafs[1]: 1})
                    c.update({i: 1})
                else:                    
                    a.update({leafs[0]: 1})
                    b.update({leafs[1]: 1})
                    c.update({i: 1})
                    
                self.list_A += [list(a.values())]
                self.list_B += [list(b.values())]
                self.list_C += [list(c.values())]

        list_A = Matrix(self.field, self.list_A)
        list_B = Matrix(self.field, self.list_B)
        list_C = Matrix(self.field, self.list_C)
        
        
        M = [list_A, list_B, list_C]
        PolyM = []
        for m in M:
            PolyList = []
            for i in range(m.ncols()):
                points = []
                for j in range(m.nrows()):
                    points.append([j+1,m[j,i]])
                    
                Poly = self.field.lagrange_polynomial(points).coefficients(sparse=False)

                if(len(Poly) < m.nrows()):
                    dif = m.nrows() - len(Poly)
                    for c in range(dif):
                        Poly.append(0);

                PolyList.append(Poly)

            PolyM.append(Matrix(self.field, PolyList))
        
        self.list_A = PolyM[0]
        self.list_B = PolyM[1]
        self.list_C = PolyM[2]
        
    def get_func_P(self):
        return self.func_P
       
    def get_H_T(self, lst_cord=None):
        #if lst_cord is None:
        #    cord = vector(self.field, [1] + [i for i in self.dict_qap_with_value.values()])
        #else:
        #old_lst_mult = self.list_multipliers.copy()
        self.get_value_qap(lst_cord)
        self.refresh_func_P()

        Z = 1
        for i in range(1, len(self.list_A[0]) + 1):
            Z *= self.field(self.monom - i)

        H = self.func_P.quo_rem(Z)
        return H[0], Z
        
        
    def init_qap(self, str_func):
        str_func = str_func.replace(' ', '')
        self.set_all_multipliers(str_func)
        lst = self.get_indexes_internal_func(str_func)

        while lst is not None:
            temp_func = self.reduce_func(str_func[lst[0] + 1:lst[1] - 1], '*')
            temp_func = self.reduce_func(temp_func, '+')
            str_func = str_func.replace(str_func[lst[0]:lst[1]], temp_func)
            lst = self.get_indexes_internal_func(str_func)

        str_func = self.reduce_func(str_func, '*')
        str_func = self.reduce_func(str_func, '+')

from sympy import sympify, Poly

def ___check_y(y):
    try:
        ret = y.sqrt(extend=False)
        return ret
    except:
        return None
    
def get_point(C, x):
    K = C.base_ring()
    f, h = C.hyperelliptic_polynomials()
    b = -h/2 if h != 0 else 0
    D = b*b + f
    _x = K(x)
    Dval = D(_x)
    y = ___check_y(Dval)
    
    print(y, _x, K)

    if not Dval: # y = 0
        return list(C.point([x, Dval, K(1)], check=True))[:2]
    elif y is not None:
        return list(C.point([x, y, K(1)], check=True))[:2]
    else:
        return None

def gen_point(C, x):
    print(C.base_ring())
    c_x = x
    p_x = get_point(C, x)
    while p_x is None:
        c_x += 1
        p_x = get_point(C, c_x)
    return p_x

def extended_gcd(u1, u2):
    if u2 == 0:
        return (1, 0)
    else:
        (e1, e2) = extended_gcd(u2, u1.quo_rem(u2)[1])
        return (e2, e1 - u1.quo_rem(u2)[0] * e2)

def enhanced_cantor(D1, D2, D):
    gcd_d = gcd(D1.u, D2.u)
    e1, e2 = extended_gcd(D1.u, D2.u)
        
    gcd_D = gcd(gcd_d, D1.v + D2.v + D1.H)
    c1, c2 = extended_gcd(gcd_d, (D1.v + D2.v + D1.H))
        
    s1 = c1 * e1
    s2 = c1 * e2
    s3 = c2
        
    u = (D1.u * D2.u).quo_rem(gcd_D)[0]
    v = ((s1 * D1.u * D2.v + s2 * D1.v * D2.u + s3 * (D1.v * D2.v + D1.F))).quo_rem(gcd_D)[0]
    rem = v.quo_rem(u)
        
    if rem[1] != 0:
        v = rem[1]
            
    deg_u = u.degree()
    f = D1.field(gcd_D.quo_rem(D.u)[1])
    g = D1.field(1)
    h = D1.field(1)
    while deg_u > D.g:
        _u = (D1.F - v * D1.H - v * v).quo_rem(u)[0]
        _v = (- D1.H - v)
        rem = _v.quo_rem(_u)
        if rem[1] != 0:
            _v = rem[1]
            
        f = (f * (D.v - v)).quo_rem(D.u)[1]
        #rem = f.quo_rem(D.u)
        #if rem[1] != 0:
        #    f = rem[1]
            
        g = g * _u
        rem = g.quo_rem(D.u)
        if rem[1] != 0:
            g = rem[1]
        
        if v.degree() > D.g:
            h = -1 * h * v.coefficient({D.monom: v.degree()}) #v.coefficients()[-1]
        u = _u
        v = _v
        deg_u = u.degree()
    
    ret_D = Divisor(D.F, D.H, D.field, D.monom)
    ret_D.init_by_u_v(u, v)
    return ret_D, f, g, h
    
def millers_alg(D1, D2, d, s):
    binary_s = bin(s)
    lst_binary_s = list(binary_s)[2:][::-1]
    N = binary_s.count('1')
    D = D1.copy()
    f1, f2, f3 = D1.field(1), D1.field(1), D1.field(1)
    
    for i in range(N-1, -1, -1):
        print("-f1 >>  ", f1)
        f1 = (f1 * f1).quo_rem(D2.u)[1]
        print("-f1 >>  ", f1, D2.u, (f1 * f1).quo_rem(D2.u))
        f2 = (f2 * f2).quo_rem(D2.u)[1]
        f3 = (f3 * f3)
        D, h1, h2, h3 = enhanced_cantor(D.copy(), D.copy(), D2.copy())
        f1 = (f1 * h1).quo_rem(D2.u)[1]
        f2 = (f2 * h2).quo_rem(D2.u)[1]
        f3 = (f3 * h3)
        if lst_binary_s[i] == '1':
            D, h1, h2, h3 = enhanced_cantor(D.copy(), D1.copy(), D2.copy())
            print("--f1 h1>>  ", f1, h1)
            f1 = (f1 * h1).quo_rem(D2.u)[1]
            print("--f1 >>  ", f1, D2.u, (f1 * f1).quo_rem(D2.u))
            f2 = (f2 * h2).quo_rem(D2.u)[1]
            f3 = (f3 * h3)

    print("f1.resultant(D2.u) >>  ", f1, D2.u, f1.resultant(D2.u))
    print("f2.resultant(D2.u) >>  ", f2, D2.u, f2.resultant(D2.u))

    res_f1 = f1.resultant(D2.u)
    res_f2 = f2.resultant(D2.u)
    return (res_f1.quo_rem(f3**D2.u.degree() * (1 if res_f2 == 0 else res_f2))[0])**d


def get_degree(str_func):
    return Poly(sympify(str_func), domain='QQ').degree()

class Divisor:
    def __init__(self, f, h, field, monom):
        self.monom = monom
        self.field = field
        self.F, self.H = f, h
        self.g = (self.F.degree() - 1) // 2
        self.lst_P = []
        self.u = 0
        self.v = 0
        
    def copy(self):
        ret_D = Divisor(self.F, self.H, self.field, self.monom)
        ret_D.init_by_u_v(self.u, self.v)
        return ret_D
    
    def init_by_lst_P(self, lst_P):
        self.lst_P = lst_P
        self.u = prod([(self.monom - i[0]) for i in lst_P])
        self.v = 0
        for i in range(0, len(lst_P)):
            self.v += prod([self.monom - lst_P[j][0] for j in range(0, len(lst_P)) if i != j]) * lst_P[i][1] / prod([lst_P[i][0] - lst_P[j][0] for j in range(0, len(lst_P)) if i != j])
                
    def init_by_u_v(self, u, v):
        self.lst_P = []
        self.u = u
        self.v = v
    
    def weil_paining(self, other_divisor, s):
        return self.field(millers_alg(self, other_divisor, 1, s))
    
    def __str__(self):
        str_lst_P = [f'P({i[0]}:{i[1]})' for i in self.lst_P]
        ret_lst_P = " + ".join(str_lst_P)
        ret_div = ret_lst_P + f" - {len(self.lst_P)} * P_inf"
        return f'Divisor = {ret_div}\n\tu = {self.u}\n\tv = {self.v}'
    
    def __add__(self, other_divisor):
        print(self, other_divisor)
        gcd_d = gcd(self.u, other_divisor.u)
        print(gcd_d)
        gcd_d = (gcd_d)
        e1, e2 = extended_gcd(self.u, other_divisor.u)
        gcd_D = gcd(gcd_d, self.v + other_divisor.v + self.H)
        print(gcd_D)
        c1, c2 = extended_gcd(gcd_d, (self.v + other_divisor.v + self.H))

        s1 = (c1 * e1)
        s2 = (c1 * e2)
        s3 = (c2)
        
        u = ((self.u * other_divisor.u).quo_rem(gcd_D**2)[0])
        v = ((s1 * self.u * other_divisor.v + s2 * self.v * other_divisor.u + s3 * (self.v * other_divisor.v + self.F))).quo_rem(gcd_D)[0]
        rem = v.quo_rem(u)
        
        if rem[1] != 0:
            v = rem[1]
            
        deg_u = u.degree()
        while deg_u > self.g:
            u = (self.F - v * self.H - v**2).quo_rem(u)[0]
            v = -1 * (self.H + v)
            print(type(u), u)
            print(format_polynomial(u.__str__()))
            print(v)
            rem = v.quo_rem(u)
            #if rem[1] != 0:
            v = rem[1]
            
            print(rem)
            print(deg_u)
            print(self.monom)
            
            #str_u = u.__str__()
            deg_str_u = -1
            for i in re.findall(r'((\w+)|(\w+_\d+))\^(\d+)?', u.__str__()):
                if i[1] == self.monom.__str__():
                    deg_str_u = deg_str_u if int(i[3]) <= deg_str_u else int(i[3])
            #if str_u[0] != '(':
            #    deg_u = u.degree(self.monom)
            deg_u = deg_str_u
        
        ret_D = Divisor(self.F, self.H, self.field, self.monom)
        ret_D.init_by_u_v(u, v)
        return ret_D

import random
import time

class snark:
    def __init__(self, p, k, r, a, func):
        self.p = p
        self.k = k
        self.r = r
        self.a = a
        self.str_func = func

        R.<x, y> = PolynomialRing(GF(p))
        self.R = R
        f_x = x^5 + a * x
        h_x = 0

        R_k.<w, q> = PolynomialRing(GF(int(p^k)))
        self.R_k = R_k
        f_w = w^5 + a * w
        h_w = 0

        P.<b> = PolynomialRing(GF(p))
        self.P = P
        f_u = b^5 + a * b
        h_u = 0

        P_k.<d> = PolynomialRing(GF(p^k))
        self.P_k = P_k
        f_v = d^5 + a * d
        h_v = 0
        
        self.qap = QAP(self.P, b)
        self.str_func = func

        self.qap.init_qap(format_polynomial(self.str_func))
        qap_value = self.qap.get_value_qap()
        self.qap.build_qap()

        H, T = self.qap.get_H_T()
        H, T = self.qap.get_H_T([3, 5, 9, 27, 30, 35])

        self.func_P = self.qap.get_func_P()
        func_p_j = 0
        var('j')
        for i in range(0, len(self.func_P.list())):
            func_p_j += int(self.func_P.list()[i])*j^i

        self.solve_func_P = [item[0] for item in solve_mod(func_p_j, p) if item[0] != 1]    
    
    def __str__(self):
        print(self.qap)
        
        print("|> CRS   ")
        print(f'| | crs[0]                 >>  {self.CRS[0]}')
        print(f'| | crs[1]                 >>  {self.CRS[1]}')        
        return '|' + '_'*100
    
    def gen_CRS(self):
        s_i = random.choice(self.solve_func_P)

        self.g = random_prime(1000)
        self.h = random_prime(1000)
        self.alpha = randint(123, 1000)
        
        self.CRS = [[self.g * s_i**i for i in range(0, len(self.func_P.list()) + 1)],
                   [self.h * self.alpha * s_i**i for i in range(0, len(self.func_P.list()) + 1)]]
        return self.CRS
    
    def gen_proof(self, crs):
        a = sum(crs[0])
        b = sum(crs[1])
        
        return a, b
    
    def verify_proof(self, a, b, crs):
        return ([a, len(crs[1])*crs[1][0]], [len(crs[0])*crs[0][0], b])
        

p = 13337
k = 4
r = 13
a = 2 

start_time_all = time.time()
_snark = snark(p, k, r, a, 'x^1000 + x + 5')

start_time = time.time() 
crs = _snark.gen_CRS()
a, b = _snark.gen_proof(crs)
proof = _snark.verify_proof(a, b, crs)

end_time = time.time()
execution_time = end_time - start_time
print("Время выполнения полное: {:.6f} секунд".format(end_time - start_time_all))
print("Время генерации доказательства: {:.6f} секунд".format(end_time - start_time))

print(proof)
print(_snark)
